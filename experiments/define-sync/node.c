#include "contiki.h"
#include "net/netstack.h"
#include "net/linkaddr.h"
#include "net/mac/tsch/tsch.h"
#include "net/mac/tsch/tsch-queue.h"
#include "net/mac/tsch/tsch-schedule.h"
#include "net/routing/routing.h"
#include "net/routing/rpl-lite/rpl.h"
#include "net/routing/rpl-lite/rpl-neighbor.h"
#include "net/ipv6/uip-ds6-nbr.h"
#include "net/link-stats.h"
#include "sys/energest.h"
#include "sys/node-id.h"
#include "sys/log.h"
#include "lib/random.h"
#include "node.h"

#define LOG_MODULE "DEFINE-NODE"
#define LOG_LEVEL LOG_LEVEL_INFO

static struct ctimer sf_timer;
static struct tsch_slotframe *sf_min;
static uint8_t join_logged = 0;
static uint16_t parent_switches = 0;
static uint32_t eb_tx_counter = 0;
static uint32_t slot_callbacks = 0;
static uint32_t boot_asn = 0;

/* Real Energest baseline snapshots taken at join instant */
static uint64_t join_tx_ticks = 0;
static uint64_t join_rx_ticks = 0;
static uint64_t boot_rx_ticks = 0;

/* DEFINE 10-byte static .bss state + 16-byte .rodata lookup table */
static uint32_t join_asn = 0;
static uint16_t prev_rank = 0;
static uint8_t ch_offset_u = 0;
static uint8_t parent_ch_offset = 0;
static uint8_t phi_u = 0;
static uint8_t sync_wait_cnt = 0;
static uint8_t parent_switched_flag = 0;
static uint32_t last_sync_asn = 0;
static uint8_t cached_h_u = 0;

/* 16-entry uint8 lookup table for exp(-gamma) scaled by 255 (P_min = 0.15 -> 38/255) */
static const uint8_t exp_lut[16] = {
  255, 199, 155, 120, 94, 73, 57, 44, 38, 38, 38, 38, 38, 38, 38, 38
};

static uint16_t
isqrt_u16(uint16_t val)
{
  uint16_t res = 0;
  uint16_t bit = 1U << 14;
  while(bit > val) {
    bit >>= 2;
  }
  while(bit != 0) {
    if(val >= res + bit) {
      val -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}

static uint16_t
get_preferred_parent_id(void)
{
  if(curr_instance.used && curr_instance.dag.preferred_parent != NULL) {
    const linkaddr_t *ll = rpl_neighbor_get_lladdr(curr_instance.dag.preferred_parent);
    if(ll != NULL) {
      return (uint16_t)ll->u8[LINKADDR_SIZE - 1];
    }
  }
  return 0;
}

static uint32_t
get_unicast_tx_retries(void)
{
  uint32_t retries = 0;
  uip_ds6_nbr_t *nbr = uip_ds6_nbr_head();
  while(nbr != NULL) {
    const linkaddr_t *ll = uip_ds6_nbr_get_ll(nbr);
    const struct link_stats *stats = link_stats_from_lladdr(ll);
    if(stats != NULL && stats->etx > 128) {
      retries += (uint32_t)((stats->etx - 128) / 64);
    }
    nbr = uip_ds6_nbr_next(nbr);
  }
  return retries;
}

void
define_on_new_time_source(const struct tsch_neighbor *old, const struct tsch_neighbor *new)
{
  if(new != NULL) {
    last_sync_asn = tsch_current_asn.ls4b;
    sync_wait_cnt = 0;
    parent_switched_flag = 1;
  }
  if(old != NULL && new != NULL && old != new) {
    parent_switches++;
    LOG_INFO("CSV_PARENT_SWITCH,id=%u,asn=%lu,switches=%u\n",
             (unsigned)node_id, (unsigned long)tsch_current_asn.ls4b, (unsigned)parent_switches);
  }
}

static void
set_slot0_link(uint8_t offset, uint8_t link_opts)
{
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(sf_min != NULL && !tsch_is_locked()) {
    struct tsch_link *l = tsch_schedule_get_link_by_timeslot(sf_min, 0);
    if(l != NULL) {
      l->channel_offset = offset;
      l->link_options = link_opts;
    } else {
      tsch_schedule_add_link(sf_min, link_opts, LINK_TYPE_ADVERTISING,
                             &tsch_broadcast_address, 0, offset, 1);
    }
  }
}

static void
safe_set_eb_period(clock_time_t period)
{
  clock_time_t min_period = (CLOCK_SECOND / 20 > 4) ? (CLOCK_SECOND / 20) : 4;
  if(period < min_period) {
    period = min_period;
  }
  tsch_set_eb_period(period);
}

static uint8_t
compute_adapt_q_u8(uint32_t asn_now, int nbrs, uint8_t use_age, uint8_t use_dens, uint8_t use_phase)
{
  uint32_t phase_slots = use_phase ? ((uint32_t)phi_u * (uint32_t)SF_LENGTH) : 0;
  if(node_id != 1 && asn_now < join_asn + phase_slots) {
    return 0;
  }
  uint32_t delta_asn = use_age
                         ? ((asn_now > join_asn + phase_slots) ? (asn_now - join_asn - phase_slots) : 0)
                         : (uint32_t)(4 * SF_LENGTH);
  uint16_t eff_n = use_dens ? (uint16_t)(1 + (nbrs > 0 ? nbrs : 0)) : 4;
  uint16_t sqrt_scaled = isqrt_u16((uint16_t)(eff_n * 100));
  uint32_t gamma_q2 = (delta_asn * sqrt_scaled * 4UL) / ((uint32_t)NUM_CHANNELS * (uint32_t)SF_LENGTH * 10UL);
  uint8_t lut_idx = (gamma_q2 > 15) ? 15 : (uint8_t)gamma_q2;
  return (exp_lut[lut_idx] > 38) ? exp_lut[lut_idx] : 38;
}

static void
update_rank_tier(uint8_t do_immediate_refresh, uint8_t do_guard, uint8_t use_id_hash)
{
  uint16_t rank_inc = 256;
  uint16_t cur_rank = (curr_instance.used && curr_instance.dag.rank >= rank_inc)
                        ? (uint16_t)curr_instance.dag.rank
                        : (uint16_t)((node_id == 1) ? 256 : 512);
  uint16_t diff_rank = (cur_rank > prev_rank) ? (cur_rank - prev_rank) : (prev_rank - cur_rank);

  if((do_immediate_refresh && parent_switched_flag) || prev_rank == 0 || diff_rank >= (rank_inc / 2)) {
    uint16_t raw_h = (cur_rank >= rank_inc) ? ((cur_rank / rank_inc) - 1) : 0;
    cached_h_u = (raw_h > 255) ? 255 : (uint8_t)raw_h;
    prev_rank = cur_rank;
    uint16_t pid = get_preferred_parent_id();
    if(use_id_hash) {
      parent_ch_offset = (uint8_t)(((uint32_t)pid * 2654435761UL) % NUM_CHANNELS);
    } else {
      parent_ch_offset = (cached_h_u > 0) ? (uint8_t)((cached_h_u - 1) % NUM_CHANNELS) : 0;
    }
    parent_switched_flag = 0;
  }

  if(use_id_hash) {
    ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
  } else {
    ch_offset_u = (uint8_t)(cached_h_u % NUM_CHANNELS);
  }
  if(do_guard && node_id != 1 && ch_offset_u == parent_ch_offset) {
    ch_offset_u = (uint8_t)((ch_offset_u + 1) % NUM_CHANNELS);
  }
}

static void
apply_scheme_schedule(void)
{
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(!tsch_is_associated || sf_min == NULL) {
    return;
  }

  slot_callbacks++;
  uint32_t asn_now = tsch_current_asn.ls4b;
  uint32_t sf_num = (uint32_t)(asn_now / SF_LENGTH);
  int nbrs = uip_ds6_nbr_num();
  clock_time_t base_sf_dur = (clock_time_t)(EB_MULT * SF_LENGTH * (CLOCK_SECOND / 100));
  if(base_sf_dur < 4) {
    base_sf_dur = 4;
  }

  uint8_t link_opts = LINK_OPTION_RX | LINK_OPTION_TX | LINK_OPTION_SHARED;
  uint8_t tx_prob_u8 = 255;
  uint8_t enable_keepalive = 0;
  uint8_t subslots = 1;

  switch(SYNC_SCHEME) {
    case 1:  /* DEFINE (Full) */
    case 26: /* DEFINE (Lossy MRM 85% PRR) */
    case 27: /* DEFINE (Reversed Sweep Order) */
    case 28: /* DEFINE (Poisson Arrivals) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      break;

    case 25: /* DEFINE (No Keepalive) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 0;
      break;

    case 14: /* EASE + ADAPT (Grafted, Alg. 1+KA) */
      update_rank_tier(0, 0, 1);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      break;

    case 15: /* EBMC + ADAPT (Grafted, Alg. 1+KA) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2 + (sf_num & 1));
      parent_ch_offset = (uint8_t)((get_preferred_parent_id() % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      break;

    case 16: /* ID-Hash + Guard + ADAPT (Control) */
      update_rank_tier(1, 1, 1);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      break;

    case 17: /* ASSIGN (Hysteresis Only, No Refresh/Guard, 1S) */
      update_rank_tier(0, 0, 0);
      tx_prob_u8 = 217; /* 0.85 EB/SF after keepalive */
      enable_keepalive = 1;
      break;

    case 18: /* ASSIGN (Refresh Only, No Guard, 1S) */
      update_rank_tier(1, 0, 0);
      tx_prob_u8 = 217;
      enable_keepalive = 1;
      break;

    case 19: /* ASSIGN (Guard Only, Hysteresis, 1S) */
      update_rank_tier(0, 1, 0);
      tx_prob_u8 = 217;
      enable_keepalive = 1;
      break;

    case 20: /* ASSIGN-Only (Refresh + Guard, 1S) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = 217;
      enable_keepalive = 1;
      break;

    case 21: /* ADAPT (Age Only, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 0, 0);
      break;

    case 22: /* ADAPT (Density Only, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 0, 1, 0);
      break;

    case 23: /* ADAPT (No Phase, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 0);
      break;

    case 24: /* ADAPT-Only (Full, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      break;

    case 2:  /* 6TiSCH-MC (1S) */
      ch_offset_u = 0;
      tx_prob_u8 = 255;
      break;

    case 11: /* 6TiSCH-MC (Tuned 4S) */
      ch_offset_u = 0;
      tx_prob_u8 = 64; /* 0.25 EB/SF */
      break;

    case 12: /* 6TiSCH-MC (Default 26S) */
      ch_offset_u = 0;
      tx_prob_u8 = 10; /* 0.04 EB/SF */
      break;

    case 3:  /* EDBA-SA */
      ch_offset_u = (uint8_t)(random_rand() % NUM_CHANNELS);
      tx_prob_u8 = 255;
      break;

    case 4:  /* ECV-SA */
      ch_offset_u = (uint8_t)(node_id % NUM_CHANNELS);
      tx_prob_u8 = 255;
      break;

    case 5:  /* ATP: 2 */
      ch_offset_u = (uint8_t)((sf_num % ATP_SUBSLOTS + 1) % NUM_CHANNELS);
      tx_prob_u8 = 255;
      subslots = ATP_SUBSLOTS;
      break;

    case 6:  /* TRGB */
    {
      uint8_t rgb_phase = (uint8_t)(sf_num % 3);
      uint8_t sub_band = (NUM_CHANNELS >= 3) ? (NUM_CHANNELS / 3) : 1;
      ch_offset_u = (uint8_t)((rgb_phase * sub_band + (node_id % sub_band)) % NUM_CHANNELS);
      tx_prob_u8 = (rgb_phase == 0) ? 204 : ((rgb_phase == 1) ? 102 : 26); /* (0.8, 0.4, 0.1) */
      break;
    }

    case 7:  /* CSMJ */
    {
      uint8_t csmj_be = (nbrs > 8) ? 3 : ((nbrs > 3) ? 2 : 1);
      ch_offset_u = 0;
      tx_prob_u8 = (uint8_t)(255 >> csmj_be);
      break;
    }

    case 8:  /* EASE */
      ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
      tx_prob_u8 = (uint8_t)(255 / (1 + (nbrs > 0 ? nbrs : 1)));
      if(tx_prob_u8 < 38) {
        tx_prob_u8 = 38;
      }
      break;

    case 9:  /* Pairing [19] */
    {
      uint8_t base_pair = (uint8_t)((node_id / 2) % (NUM_CHANNELS / 2));
      ch_offset_u = (sf_num & 1) ? (uint8_t)(base_pair + (NUM_CHANNELS / 2)) : base_pair;
      tx_prob_u8 = 255;
      break;
    }

    case 10: /* EBMC (1S) [20] */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = 128; /* 0.50 EB/SF (alternating EB vs DIO sub-offsets) */
      break;

    case 13: /* EBMC (Tuned 2S) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = 64;  /* 0.25 EB/SF */
      break;

    default:
      ch_offset_u = 0;
      tx_prob_u8 = 255;
      break;
  }

  if(enable_keepalive && node_id != 1) {
    uint32_t sync_age_slots = (asn_now > last_sync_asn) ? (asn_now - last_sync_asn) : 0;
    if(sync_age_slots >= 3000UL) {
      if(sync_wait_cnt < 20) {
        sync_wait_cnt++;
        set_slot0_link(parent_ch_offset, LINK_OPTION_RX | LINK_OPTION_TIME_KEEPING);
        return;
      } else {
        parent_ch_offset = (uint8_t)((parent_ch_offset + 1) % NUM_CHANNELS);
        sync_wait_cnt = 0;
        last_sync_asn = asn_now;
      }
    }
  }

  uint8_t r_draw = (uint8_t)(random_rand() & 0xFF);
  if(r_draw <= tx_prob_u8) {
    set_slot0_link(ch_offset_u, LINK_OPTION_TX);
    safe_set_eb_period(base_sf_dur);
    eb_tx_counter += subslots;
    uint16_t pid = get_preferred_parent_id();
    LOG_INFO("CSV_SLOT0_TX,id=%u,sf=%lu,ch=%u,tier=%u,parent=%u,sub=%u\n",
             (unsigned)node_id, (unsigned long)sf_num, (unsigned)ch_offset_u,
             (unsigned)cached_h_u, (unsigned)pid, (unsigned)subslots);
  } else {
    set_slot0_link(0, link_opts);
    safe_set_eb_period(base_sf_dur * 16);
  }

  if((slot_callbacks % 10) == 0) {
    energest_flush();
    uint64_t cur_tx = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    uint64_t cur_rx = energest_type_time(ENERGEST_TYPE_LISTEN);
    uint32_t slot0_tx_us = (uint32_t)(((cur_tx > join_tx_ticks ? (cur_tx - join_tx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
    uint32_t slot0_rx_us = (uint32_t)(((cur_rx > join_rx_ticks ? (cur_rx - join_rx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
    LOG_INFO("CSV_ENERGEST,id=%u,sf=%lu,eb=%lu,tx_us=%lu,rx_us=%lu,nbrs=%d,switches=%u,retries=%lu\n",
             (unsigned)node_id, (unsigned long)sf_num, (unsigned long)eb_tx_counter,
             (unsigned long)slot0_tx_us, (unsigned long)slot0_rx_us,
             nbrs, (unsigned)parent_switches, (unsigned long)get_unicast_tx_retries());
  }
}

static void
slotframe_callback(void *ptr)
{
  apply_scheme_schedule();
  ctimer_reset(&sf_timer);
}

PROCESS(define_node_process, "DEFINE 6TiSCH Experiment Node");
AUTOSTART_PROCESSES(&define_node_process);

PROCESS_THREAD(define_node_process, ev, data)
{
  static struct etimer check_timer;
  static struct etimer arrival_timer;
  PROCESS_BEGIN();

  energest_flush();
  boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);

  if(node_id == 1) {
    NETSTACK_ROUTING.root_start();
    tsch_set_coordinator(1);
    join_asn = 0;
    boot_asn = 0;
    last_sync_asn = 0;
    phi_u = 0;
    join_tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    join_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
    apply_scheme_schedule();
    join_logged = 1;
  } else {
#if POISSON_ARRIVAL
    clock_time_t arrival_delay = (clock_time_t)(((node_id - 1) * CLOCK_SECOND) / 5 + (random_rand() % (CLOCK_SECOND / 5)));
    etimer_set(&arrival_timer, arrival_delay);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&arrival_timer));
#else
    (void)arrival_timer;
#endif
    boot_asn = tsch_current_asn.ls4b;
    energest_flush();
    boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(!join_logged) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));

    if(tsch_is_associated && NETSTACK_ROUTING.node_is_joined()) {
      join_asn = tsch_current_asn.ls4b;
      last_sync_asn = join_asn;
      uint8_t max_phi = (NUM_CHANNELS < 4) ? NUM_CHANNELS : 4;
      phi_u = (uint8_t)(((node_id ^ (join_asn / SF_LENGTH)) + (random_rand() & 0x03)) % max_phi);
      parent_switched_flag = 1;

      energest_flush();
      join_tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
      join_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
      uint32_t scan_rx_us = (uint32_t)(((join_rx_ticks > boot_rx_ticks ? (join_rx_ticks - boot_rx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
      uint32_t elapsed_join_slots = (join_asn > boot_asn) ? (join_asn - boot_asn) : join_asn;

      apply_scheme_schedule();

      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,scan_rx_us=%lu,parent=%u,tier=%u,nbrs=%d\n",
               (unsigned)node_id,
               (unsigned long)elapsed_join_slots,
               (unsigned long)scan_rx_us,
               (unsigned)get_preferred_parent_id(),
               (unsigned)cached_h_u,
               uip_ds6_nbr_num());
      join_logged = 1;
    } else {
      etimer_reset(&check_timer);
    }
  }

  PROCESS_END();
}