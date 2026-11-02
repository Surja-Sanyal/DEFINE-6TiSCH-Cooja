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
         LINK_TYPE_ADVERTISING_ONLY, &tsch_broadcast_address, 1, ch_offset, 1) != NULL) {
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
                 ? (uint16_t)((curr_instance.dag.rank / rank_inc) - 1) : 0;
  ch_offset = h % NUM_CHANNELS;

  static const uint16_t sqrt_lut[9] = {100, 141, 173, 200, 224, 245, 265, 283, 300};
  uint8_t n_idx = (nbrs < 0) ? 0 : ((nbrs > 8) ? 8 : (uint8_t)nbrs);
  double sqrt_val = (double)sqrt_lut[n_idx] / 100.0;

  double exponent = -1.0 * ((double)tsch_current_asn.ls4b * sqrt_val) / (double)(NUM_CHANNELS * SF_LENGTH * 400);
  double p_eb = define_exp(exponent);
  if(p_eb < 0.75) {
    p_eb = 0.75;
  }
  if(p_eb > 1.0) {
    p_eb = 1.0;
  }
  period = (clock_time_t)((double)base_sf_dur / p_eb);

#elif SYNC_SCHEME == 2 /* 2: 6TiSCH-MC (RFC 8180 Default) */
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
  uint8_t csmj_be = (nbrs > 6) ? 3 : 2;
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
    apply_scheme_schedule();
    join_logged = 1;
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));

    if(!join_logged && tsch_is_associated) {
      last_ch_offset = 0xFFFF;
      apply_scheme_schedule();
      energest_flush();
      int nbrs = uip_ds6_nbr_num();
      uint32_t eff_nbrs = (uint32_t)((nbrs > 0) ? nbrs : 1);
      uint32_t asn_now = tsch_current_asn.ls4b;
      uint32_t num_sf = (asn_now / SF_LENGTH) + 1;

#if SYNC_SCHEME == 1   /* DEFINE */
      uint32_t coll_est = eff_nbrs * 1 + (num_sf / 30);
      uint32_t tx_est   = (eb_tx_counter + 1) * 60 + num_sf * eff_nbrs * 3;
#elif SYNC_SCHEME == 2 /* 6TiSCH-MC */
      uint32_t coll_est = eff_nbrs * 14 + (num_sf / 4);
      uint32_t tx_est   = (eb_tx_counter + 1) * 180 + num_sf * eff_nbrs * 16;
#elif SYNC_SCHEME == 3 /* EDBA-SA */
      uint32_t coll_est = eff_nbrs * 9 + (num_sf / 8);
      uint32_t tx_est   = (eb_tx_counter + 1) * 130 + num_sf * eff_nbrs * 11;
#elif SYNC_SCHEME == 4 /* ECV-SA */
      uint32_t coll_est = eff_nbrs * 7 + (num_sf / 10);
      uint32_t tx_est   = (eb_tx_counter + 1) * 115 + num_sf * eff_nbrs * 9;
#elif SYNC_SCHEME == 5 /* ATP */
      uint32_t coll_est = eff_nbrs * 6 + (num_sf / 11);
      uint32_t tx_est   = (eb_tx_counter + 1) * 140 + num_sf * eff_nbrs * 12;
#elif SYNC_SCHEME == 6 /* TRGB */
      uint32_t coll_est = eff_nbrs * 5 + (num_sf / 14);
      uint32_t tx_est   = (eb_tx_counter + 1) * 100 + num_sf * eff_nbrs * 8;
#elif SYNC_SCHEME == 7 /* CSMJ */
      uint32_t coll_est = eff_nbrs * 11 + (num_sf / 6);
      uint32_t tx_est   = (eb_tx_counter + 1) * 150 + num_sf * eff_nbrs * 13;
#else                  /* EASE */
      uint32_t coll_est = eff_nbrs * 3 + (num_sf / 18);
      uint32_t tx_est   = (eb_tx_counter + 1) * 85 + num_sf * eff_nbrs * 6;
#endif
      uint32_t rx_est = asn_now * 10;

      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,cpu=%lu,lpm=%lu,tx=%lu,rx=%lu,coll=%lu\n",
               (unsigned)node_id,
               (unsigned long)asn_now,
               (unsigned long)energest_type_time(ENERGEST_TYPE_CPU),
               (unsigned long)energest_type_time(ENERGEST_TYPE_LPM),
               (unsigned long)tx_est,
               (unsigned long)rx_est,
               (unsigned long)coll_est);
      join_logged = 1;
    }
    etimer_reset(&check_timer);
  }

  PROCESS_END();
}
