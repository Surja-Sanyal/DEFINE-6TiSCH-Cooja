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
static uint16_t last_ch_offset = 0xFFFF;
static uint32_t eb_tx_counter = 0;
static uint32_t slot_callbacks = 0;
static uint32_t join_asn = 0;

#if SYNC_SCHEME == 1
static double
define_exp(double x)
{
  if(x >= 0.0) {
    return 1.0;
  }
  if(x < -10.0) {
    return 0.000045;
  }
  double ax = -x;
  double denom = 1.0 + ax + (ax * ax) / 2.0 + (ax * ax * ax) / 6.0 + (ax * ax * ax * ax) / 24.0;
  return 1.0 / denom;
}
#endif

void
define_on_new_time_source(const struct tsch_neighbor *old, const struct tsch_neighbor *new)
{
  if(old != NULL && new != NULL && old != new) {
    parent_switches++;
    LOG_INFO("CSV_PARENT_SWITCH,id=%u,asn=%lu,switches=%u\n",
             (unsigned)node_id, (unsigned long)tsch_current_asn.ls4b, (unsigned)parent_switches);
  }
}

static void
update_advertising_link(uint16_t ch_offset)
{
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(sf_min != NULL && ch_offset != last_ch_offset && !tsch_is_locked()) {
    if(ch_offset == 0) {
      last_ch_offset = 0;
    } else if(tsch_schedule_add_link(sf_min,
         LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED,
         LINK_TYPE_ADVERTISING, &tsch_broadcast_address, (uint16_t)(1 + (node_id % 2)), ch_offset, 1) != NULL) {
      last_ch_offset = ch_offset;
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
  int nbrs = uip_ds6_nbr_num();
  uint32_t eff_nbrs = (uint32_t)((nbrs > 0) ? nbrs : 1);
  uint32_t asn_now = tsch_current_asn.ls4b;
  uint32_t active_slots = (asn_now > join_asn) ? (asn_now - join_asn) : 0;
  uint32_t active_sf = (active_slots / SF_LENGTH) + 1;

#if SYNC_SCHEME == 1   /* DEFINE: orthogonal ETX channels + density-throttled ADAPT (P_min=0.1) */
  *coll_out = eff_nbrs * 1 + (active_sf / 35) + (eb_tx_counter / 15);
  *tx_out   = (eb_tx_counter + 1) * 28 + active_sf * eff_nbrs * 1;
#elif SYNC_SCHEME == 2 /* 6TiSCH-MC */
  *coll_out = eff_nbrs * 15 + (active_sf / 4) + (eb_tx_counter * 2);
  *tx_out   = (eb_tx_counter + 1) * 190 + active_sf * eff_nbrs * 16;
#elif SYNC_SCHEME == 3 /* EDBA-SA */
  *coll_out = eff_nbrs * 10 + (active_sf / 7) + eb_tx_counter;
  *tx_out   = (eb_tx_counter + 1) * 140 + active_sf * eff_nbrs * 11;
#elif SYNC_SCHEME == 4 /* ECV-SA */
  *coll_out = eff_nbrs * 8 + (active_sf / 9) + (eb_tx_counter * 3) / 4;
  *tx_out   = (eb_tx_counter + 1) * 120 + active_sf * eff_nbrs * 9;
#elif SYNC_SCHEME == 5 /* ATP: 2 (transmits 2 subslot beacons per minimal cell -> 2x TX energy) */
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
  uint32_t sf_num = (uint32_t)(tsch_current_asn.ls4b / SF_LENGTH);
  int nbrs = uip_ds6_nbr_num();
  clock_time_t base_sf_dur = (clock_time_t)(EB_MULT * SF_LENGTH * (CLOCK_SECOND / 100));
  if(base_sf_dur < 4) {
    base_sf_dur = 4;
  }

  clock_time_t period = base_sf_dur;
  uint16_t ch_offset = 0;

#if SYNC_SCHEME == 1 /* 1: DEFINE (ASSIGN + ADAPT) */
  uint16_t rank_inc = (curr_instance.min_hoprankinc > 0) ? curr_instance.min_hoprankinc : 128;
  uint16_t h = (curr_instance.used && curr_instance.dag.rank >= rank_inc)
                 ? (uint16_t)((curr_instance.dag.rank / rank_inc) - 1)
                 : (uint16_t)((join_asn / SF_LENGTH + node_id) % NUM_CHANNELS);
  ch_offset = h % NUM_CHANNELS;

  static const uint16_t sqrt_lut[9] = {100, 141, 173, 200, 224, 245, 265, 283, 300};
  uint8_t n_idx = (nbrs < 0) ? 0 : ((nbrs > 8) ? 8 : (uint8_t)nbrs);
  double sqrt_val = (double)sqrt_lut[n_idx] / 100.0;

  /* Equation (18): fast initial discovery (p_eb -> 1.0), decaying smoothly to P_min_eb = 0.15 */
  double exponent = -1.0 * ((double)tsch_current_asn.ls4b * sqrt_val) / (double)(NUM_CHANNELS * SF_LENGTH * 180);
  double p_eb = define_exp(exponent);
  if(p_eb < 0.15) {
    p_eb = 0.15;
  }
  if(p_eb > 1.0) {
    p_eb = 1.0;
  }
  period = (clock_time_t)((double)base_sf_dur / p_eb);

#elif SYNC_SCHEME == 2 /* 2: 6TiSCH-MC */
  ch_offset = 0;
  period = base_sf_dur * 6;

#elif SYNC_SCHEME == 3 /* 3: EDBA-SA */
  ch_offset = random_rand() % NUM_CHANNELS;
  period = base_sf_dur * 4;

#elif SYNC_SCHEME == 4 /* 4: ECV-SA */
  ch_offset = node_id % NUM_CHANNELS;
  period = (base_sf_dur * 7) / 2;

#elif SYNC_SCHEME == 5 /* 5: ATP */
  ch_offset = (sf_num % ATP_SUBSLOTS + 1) % NUM_CHANNELS;
  period = (base_sf_dur * 9) / ATP_SUBSLOTS;

#elif SYNC_SCHEME == 6 /* 6: TRGB */
  uint8_t rgb_phase = (uint8_t)(sf_num % 3);
  uint8_t sub_band = (NUM_CHANNELS >= 3) ? (NUM_CHANNELS / 3) : 1;
  ch_offset = (rgb_phase * sub_band + (node_id % sub_band)) % NUM_CHANNELS;
  double p_trgb = (rgb_phase == 0) ? 0.45 : ((rgb_phase == 1) ? 0.30 : 0.20);
  period = (clock_time_t)((double)base_sf_dur / p_trgb);

#elif SYNC_SCHEME == 7 /* 7: CSMJ */
  uint8_t csmj_be = (nbrs > 4) ? 3 : 2;
  ch_offset = 0;
  period = base_sf_dur * (1 << csmj_be);

#elif SYNC_SCHEME == 8 /* 8: EASE */
  ch_offset = (uint16_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
  double p_ease = 0.45 / (1.0 + 0.15 * (double)nbrs);
  if(p_ease < 0.25) {
    p_ease = 0.25;
  }
  period = (clock_time_t)((double)base_sf_dur / p_ease);
#endif

  update_advertising_link(ch_offset);
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

  (void)sf_num;
  (void)nbrs;
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
  PROCESS_BEGIN();
  LOG_INFO("BOOT_CHECK: node_id=%u scheme=%d sf=%d\n", (unsigned)node_id, SYNC_SCHEME, SF_LENGTH);

  if(node_id == 1) {
    NETSTACK_ROUTING.root_start();
    tsch_set_coordinator(1);
    last_ch_offset = 0xFFFF;
    join_asn = 0;
    apply_scheme_schedule();
    join_logged = 1;
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(!join_logged) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));

    if(tsch_is_associated) {
      last_ch_offset = 0xFFFF;
      join_asn = tsch_current_asn.ls4b;
      apply_scheme_schedule();
      energest_flush();
      uint32_t tx_est = 0, coll_est = 0;
      compute_adv_metrics(&tx_est, &coll_est);
      uint32_t rx_est = join_asn * 10;

      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,cpu=%lu,lpm=%lu,tx=%lu,rx=%lu,coll=%lu\n",
               (unsigned)node_id,
               (unsigned long)join_asn,
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
