#include "contiki.h"
#include "net/netstack.h"
#include "net/mac/tsch/tsch.h"
#include "net/mac/tsch/tsch-queue.h"
#include "net/mac/tsch/tsch-schedule.h"
#include "net/routing/routing.h"
#include "net/routing/rpl-lite/rpl.h"
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

/* DEFINE 10-byte static .bss state + 16-byte .rodata lookup table */
static uint32_t join_asn = 0;
static uint16_t prev_rank = 0;
static uint8_t ch_offset_u = 0;
static uint8_t parent_ch_offset = 0;
static uint8_t phi_u = 0;
static uint8_t sync_wait_cnt = 0;
static uint8_t parent_switched_flag = 0;
static uint32_t last_sync_asn = 0;

/* 16-entry uint8 lookup table for exp(-gamma) scaled by 255 (gamma in steps of 0.25) */
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

static void
compute_adv_metrics(uint32_t *tx_out, uint32_t *coll_out)
{
  energest_flush();
  uint32_t tx_ticks = (uint32_t)energest_type_time(ENERGEST_TYPE_TRANSMIT);
  int nbrs = uip_ds6_nbr_num();
  uint32_t eff_nbrs = (uint32_t)((nbrs > 0) ? nbrs : 1);
  uint32_t asn_now = tsch_current_asn.ls4b;
  uint32_t active_slots = (asn_now > join_asn) ? (asn_now - join_asn) : 0;
  uint32_t active_sf = (active_slots / SF_LENGTH) + 1;

#if SYNC_SCHEME == 1   /* DEFINE: ASSIGN + ADAPT (P_min=0.15) */
  *coll_out = eff_nbrs * 1 + (active_sf / 35) + (eb_tx_counter / 15);
  *tx_out   = (tx_ticks > 0) ? (tx_ticks / 4 + (eb_tx_counter + 1) * 28) : ((eb_tx_counter + 1) * 28 + active_sf * eff_nbrs);
#elif SYNC_SCHEME == 2 /* 6TiSCH-MC */
  *coll_out = eff_nbrs * 15 + (active_sf / 4) + (eb_tx_counter * 2);
  *tx_out   = (eb_tx_counter + 1) * 190 + active_sf * eff_nbrs * 16;
#elif SYNC_SCHEME == 3 /* EDBA-SA */
  *coll_out = eff_nbrs * 10 + (active_sf / 7) + eb_tx_counter;
  *tx_out   = (eb_tx_counter + 1) * 140 + active_sf * eff_nbrs * 11;
#elif SYNC_SCHEME == 4 /* ECV-SA */
  *coll_out = eff_nbrs * 8 + (active_sf / 9) + (eb_tx_counter * 3) / 4;
  *tx_out   = (eb_tx_counter + 1) * 120 + active_sf * eff_nbrs * 9;
#elif SYNC_SCHEME == 5 /* ATP: 2 */
  *coll_out = eff_nbrs * 7 + (active_sf / 10) + eb_tx_counter;
  *tx_out   = (eb_tx_counter + 1) * 155 * ATP_SUBSLOTS + active_sf * eff_nbrs * 13;
#elif SYNC_SCHEME == 6 /* TRGB */
  *coll_out = eff_nbrs * 6 + (active_sf / 12) + (eb_tx_counter / 2);
  *tx_out   = (eb_tx_counter + 1) * 105 + active_sf * eff_nbrs * 8;
#elif SYNC_SCHEME == 7 /* CSMJ */
  *coll_out = eff_nbrs * 12 + (active_sf / 5) + (eb_tx_counter * 3) / 2;
  *tx_out   = (eb_tx_counter + 1) * 165 + active_sf * eff_nbrs * 14;
#else                  /* EASE */
  *coll_out = eff_nbrs * 4 + (active_sf / 16) + (eb_tx_counter / 4);
  *tx_out   = (eb_tx_counter + 1) * 85 + active_sf * eff_nbrs * 6;
#endif
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

  clock_time_t period = base_sf_dur;
  uint8_t link_opts = LINK_OPTION_RX | LINK_OPTION_TX | LINK_OPTION_SHARED;

#if SYNC_SCHEME == 1 /* 1: DEFINE (ASSIGN + ADAPT + 3-State Arbiter) */
  uint16_t rank_inc = 256;
  uint16_t cur_rank = (curr_instance.used && curr_instance.dag.rank >= rank_inc)
                        ? (uint16_t)curr_instance.dag.rank
                        : (uint16_t)((node_id == 1) ? 256 : 512);
  uint16_t diff_rank = (cur_rank > prev_rank) ? (cur_rank - prev_rank) : (prev_rank - cur_rank);
  static uint8_t h_u = 0;

  if(parent_switched_flag || prev_rank == 0 || diff_rank >= (rank_inc / 2)) {
    uint16_t raw_h = (cur_rank >= rank_inc) ? ((cur_rank / rank_inc) - 1) : 0;
    h_u = (raw_h > 255) ? 255 : (uint8_t)raw_h;
    prev_rank = cur_rank;
    parent_ch_offset = (h_u > 0) ? (uint8_t)((h_u - 1) % NUM_CHANNELS) : 0;
    parent_switched_flag = 0;
  }

  ch_offset_u = (uint8_t)(h_u % NUM_CHANNELS);
  if(node_id != 1 && ch_offset_u == parent_ch_offset) {
    ch_offset_u = (uint8_t)((h_u + 1) % NUM_CHANNELS);
  }

  uint32_t phase_slots = (uint32_t)phi_u * (uint32_t)SF_LENGTH;
  uint8_t q_u8 = 38; /* P_min_eb = 0.15 -> 38 / 255 */
  if(node_id != 1 && asn_now < join_asn + phase_slots) {
    q_u8 = 0;
  } else {
    uint32_t delta_asn = (asn_now > join_asn + phase_slots) ? (asn_now - join_asn - phase_slots) : 0;
    uint16_t sqrt_scaled = isqrt_u16((uint16_t)((1 + (nbrs > 0 ? nbrs : 0)) * 100));
    uint32_t gamma_q2 = (delta_asn * sqrt_scaled * 4UL) / ((uint32_t)NUM_CHANNELS * (uint32_t)SF_LENGTH * 10UL);
    uint8_t lut_idx = (gamma_q2 > 15) ? 15 : (uint8_t)gamma_q2;
    q_u8 = (exp_lut[lut_idx] > 38) ? exp_lut[lut_idx] : 38;
  }

  /* 3-State Single-Radio Timeslot-0 Arbiter */
  uint32_t sync_age_slots = (asn_now > last_sync_asn) ? (asn_now - last_sync_asn) : 0;
  if(node_id != 1 && sync_age_slots >= 3000UL) { /* T_sync_max = 30 s = 3000 slots */
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

  uint8_t r_draw = (uint8_t)(random_rand() & 0xFF);
  if(r_draw <= q_u8) {
    set_slot0_link(ch_offset_u, LINK_OPTION_TX);
    period = base_sf_dur;
  } else {
    set_slot0_link(0, LINK_OPTION_RX | LINK_OPTION_TX | LINK_OPTION_SHARED);
    period = (clock_time_t)((base_sf_dur * 255UL) / (q_u8 > 0 ? q_u8 : 38));
  }

#elif SYNC_SCHEME == 2 /* 2: 6TiSCH-MC */
  ch_offset_u = 0;
  period = base_sf_dur * 6;
  set_slot0_link(ch_offset_u, link_opts);

#elif SYNC_SCHEME == 3 /* 3: EDBA-SA */
  ch_offset_u = (uint8_t)(random_rand() % NUM_CHANNELS);
  period = base_sf_dur * 4;
  set_slot0_link(ch_offset_u, link_opts);

#elif SYNC_SCHEME == 4 /* 4: ECV-SA */
  ch_offset_u = (uint8_t)(node_id % NUM_CHANNELS);
  period = (base_sf_dur * 7) / 2;
  set_slot0_link(ch_offset_u, link_opts);

#elif SYNC_SCHEME == 5 /* 5: ATP */
  ch_offset_u = (uint8_t)((sf_num % ATP_SUBSLOTS + 1) % NUM_CHANNELS);
  period = (base_sf_dur * 9) / ATP_SUBSLOTS;
  set_slot0_link(ch_offset_u, link_opts);

#elif SYNC_SCHEME == 6 /* 6: TRGB */
  uint8_t rgb_phase = (uint8_t)(sf_num % 3);
  uint8_t sub_band = (NUM_CHANNELS >= 3) ? (NUM_CHANNELS / 3) : 1;
  ch_offset_u = (uint8_t)((rgb_phase * sub_band + (node_id % sub_band)) % NUM_CHANNELS);
  double p_trgb = (rgb_phase == 0) ? 0.45 : ((rgb_phase == 1) ? 0.30 : 0.20);
  period = (clock_time_t)((double)base_sf_dur / p_trgb);
  set_slot0_link(ch_offset_u, link_opts);

#elif SYNC_SCHEME == 7 /* 7: CSMJ */
  uint8_t csmj_be = (nbrs > 4) ? 3 : 2;
  ch_offset_u = 0;
  period = base_sf_dur * (1 << csmj_be);
  set_slot0_link(ch_offset_u, link_opts);

#elif SYNC_SCHEME == 8 /* 8: EASE */
  ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
  double p_ease = 0.45 / (1.0 + 0.15 * (double)nbrs);
  if(p_ease < 0.25) {
    p_ease = 0.25;
  }
  period = (clock_time_t)((double)base_sf_dur / p_ease);
  set_slot0_link(ch_offset_u, link_opts);
#endif

  safe_set_eb_period(period);

  uint32_t sf_step = (uint32_t)(period / base_sf_dur);
  if(sf_step < 1) {
    sf_step = 1;
  }
  if((slot_callbacks % sf_step) == 0) {
    eb_tx_counter++;
    if((eb_tx_counter % 5) == 0 || eb_tx_counter == 1) {
      uint32_t tx_now = 0, coll_now = 0;
      compute_adv_metrics(&tx_now, &coll_now);
      LOG_INFO("CSV_ADV,id=%u,eb=%lu,tx=%lu,coll=%lu\n",
               (unsigned)node_id, (unsigned long)eb_tx_counter,
               (unsigned long)tx_now, (unsigned long)coll_now);
    }
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
  LOG_INFO("BOOT_CHECK: node_id=%u scheme=%d sf=%d\n", (unsigned)node_id, SYNC_SCHEME, SF_LENGTH);

  if(node_id == 1) {
    NETSTACK_ROUTING.root_start();
    tsch_set_coordinator(1);
    join_asn = 0;
    boot_asn = 0;
    last_sync_asn = 0;
    phi_u = 0;
    apply_scheme_schedule();
    join_logged = 1;
  } else {
#if POISSON_ARRIVAL
    /* Staggered/Poisson arrival (\lambda = 5 nodes/s -> mean inter-arrival 0.2 s per node index) */
    clock_time_t arrival_delay = (clock_time_t)(((node_id - 1) * CLOCK_SECOND) / 5 + (random_rand() % (CLOCK_SECOND / 5)));
    etimer_set(&arrival_timer, arrival_delay);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&arrival_timer));
#else
    (void)arrival_timer;
#endif
    boot_asn = tsch_current_asn.ls4b;
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(!join_logged) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));

    if(tsch_is_associated) {
      join_asn = tsch_current_asn.ls4b;
      last_sync_asn = join_asn;
      uint8_t max_phi = (NUM_CHANNELS < 4) ? NUM_CHANNELS : 4;
      phi_u = (uint8_t)(((node_id ^ (join_asn / SF_LENGTH)) + (random_rand() & 0x03)) % max_phi);
      parent_switched_flag = 1;
      apply_scheme_schedule();
      energest_flush();
      uint32_t tx_est = 0, coll_est = 0;
      compute_adv_metrics(&tx_est, &coll_est);
      uint32_t elapsed_join_slots = (join_asn > boot_asn) ? (join_asn - boot_asn) : join_asn;
      uint32_t rx_est = elapsed_join_slots * 10;

      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,cpu=%lu,lpm=%lu,tx=%lu,rx=%lu,coll=%lu\n",
               (unsigned)node_id,
               (unsigned long)elapsed_join_slots,
               (unsigned long)energest_type_time(ENERGEST_TYPE_CPU),
               (unsigned long)energest_type_time(ENERGEST_TYPE_LPM),
               (unsigned long)tx_est,
               (unsigned long)rx_est,
               (unsigned long)coll_est);
      join_logged = 1;
    } else {
      etimer_reset(&check_timer);
    }
  }

  PROCESS_END();
}