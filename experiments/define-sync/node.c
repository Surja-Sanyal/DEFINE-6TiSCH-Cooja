#include "contiki.h"
#include "net/netstack.h"
#include "net/mac/tsch/tsch.h"
#include "net/routing/routing.h"
#include "net/routing/rpl-lite/rpl.h"
#include "net/ipv6/uip-ds6-nbr.h"
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

/* Fast inline exponential approximation for exp(x) where x <= 0 */
static double
define_exp(double x)
{
  if(x < -10.0) {
    return 0.000045;
  }
  if(x > 0.0) {
    return 1.0;
  }
  /* Taylor series expansion: 1 + x + x^2/2 + x^3/6 + x^4/24 */
  double xx = x;
  return 1.0 + xx + (xx * xx) / 2.0 + (xx * xx * xx) / 6.0 + (xx * xx * xx * xx) / 24.0;
}

/* Callback declared by TSCH_CALLBACK_NEW_TIME_SOURCE in project-conf.h */
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
slotframe_callback(void *ptr)
{
  if(tsch_is_associated && sf_min != NULL) {
    uint32_t sf_num = (uint32_t)(tsch_current_asn.ls4b / SF_LENGTH);
    int nbrs = uip_ds6_nbr_num();
    clock_time_t base_sf_dur = (clock_time_t)(EB_MULT * SF_LENGTH * (CLOCK_SECOND / 100));

    (void)sf_num;
    (void)nbrs;
    (void)base_sf_dur;

#if SYNC_SCHEME == 1 /* 1: DEFINE (ASSIGN + ADAPT) */
    uint16_t rank_inc = (curr_instance.min_hoprankinc > 0) ? curr_instance.min_hoprankinc : 128;
    uint16_t h = curr_instance.used ? (uint16_t)(curr_instance.dag.rank / rank_inc) : 0;
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, h % NUM_CHANNELS, 1);

    /* Fast LUT for 100 * sqrt(1 + nbrs) to avoid soft-float sqrt() overhead */
    static const uint16_t sqrt_lut[9] = {100, 141, 173, 200, 224, 245, 265, 283, 300};
    uint8_t n_idx = (nbrs < 0) ? 0 : ((nbrs > 8) ? 8 : (uint8_t)nbrs);
    double sqrt_val = (double)sqrt_lut[n_idx] / 100.0;

    double exponent = -1.0 * ((double)tsch_current_asn.ls4b * sqrt_val) / (double)(NUM_CHANNELS * SF_LENGTH);
    double p_eb = define_exp(exponent);
    if(p_eb < 0.01) {
      p_eb = 0.01;
    }
    clock_time_t period = (clock_time_t)((double)base_sf_dur / p_eb);
    tsch_set_eb_period(period);

#elif SYNC_SCHEME == 2 /* 2: 6TiSCH-MC (RFC 8180 Default) */
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, 0, 1);
    tsch_set_eb_period(base_sf_dur * 10);

#elif SYNC_SCHEME == 3 /* 3: EDBA-SA (Random Channel Selection in Minimal Cell) */
    uint16_t rand_ch = random_rand() % NUM_CHANNELS;
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, rand_ch, 1);
    tsch_set_eb_period(base_sf_dur);

#elif SYNC_SCHEME == 4 /* 4: ECV-SA (Deterministic Channel Offset by Node ID) */
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, node_id % NUM_CHANNELS, 1);
    tsch_set_eb_period(base_sf_dur);

#elif SYNC_SCHEME == 5 /* 5: ATP (Multi-Subslot Frequency Rotating Advertisement) */
    uint16_t subslot_ch = (sf_num % ATP_SUBSLOTS + 1) % NUM_CHANNELS;
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, subslot_ch, 1);
    tsch_set_eb_period((base_sf_dur * 10) / ATP_SUBSLOTS);

#elif SYNC_SCHEME == 6 /* 6: TRGB (Time-Variant RGB Model - Kalita & Khatua, 2023) */
    uint8_t rgb_phase = (uint8_t)(sf_num % 3);
    uint8_t sub_band = (NUM_CHANNELS >= 3) ? (NUM_CHANNELS / 3) : 1;
    uint16_t trgb_ch = (rgb_phase * sub_band + (node_id % sub_band)) % NUM_CHANNELS;
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, trgb_ch, 1);
    double p_trgb = (rgb_phase == 0) ? 0.80 : ((rgb_phase == 1) ? 0.40 : 0.10);
    tsch_set_eb_period((clock_time_t)((double)base_sf_dur / p_trgb));

#elif SYNC_SCHEME == 7 /* 7: CSMJ (Comprehensive 6TiSCH Sync Model - Chang et al., 2026) */
    uint8_t csmj_be = (nbrs > 8) ? 4 : ((nbrs > 4) ? 3 : ((nbrs > 2) ? 2 : 1));
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, 0, 1);
    tsch_set_eb_period(base_sf_dur * (1 << csmj_be));

#elif SYNC_SCHEME == 8 /* 8: EASE (Autonomous Energy-Aware Cell Scheduling - Kalita, 2026) */
    uint16_t ease_ch = (uint16_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
    tsch_schedule_add_link(sf_min,
      LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING,
      LINK_TYPE_ADVERTISING, &tsch_broadcast_address, 0, ease_ch, 1);
    double p_ease = 0.10 / (1.0 + 0.5 * (double)nbrs);
    if(p_ease < 0.02) {
      p_ease = 0.02;
    }
    tsch_set_eb_period((clock_time_t)((double)base_sf_dur / p_ease));
#endif
  }
  ctimer_reset(&sf_timer);
}

PROCESS(define_node_process, "DEFINE 6TiSCH Experiment Node");
AUTOSTART_PROCESSES(&define_node_process);

PROCESS_THREAD(define_node_process, ev, data)
{
  static struct etimer check_timer;
  PROCESS_BEGIN();

  sf_min = tsch_schedule_get_slotframe_by_handle(0);

  /* Node 1 acts as the 6TiSCH PAN Coordinator & RPL Root */
  if(node_id == 1) {
    NETSTACK_ROUTING.root_start();
    tsch_set_coordinator(1);
    join_logged = 1;
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));

    if(!join_logged && tsch_is_associated && NETSTACK_ROUTING.node_is_reachable()) {
      energest_flush();
      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,cpu=%lu,lpm=%lu,tx=%lu,rx=%lu\n",
               (unsigned)node_id,
               (unsigned long)tsch_current_asn.ls4b,
               (unsigned long)energest_type_time(ENERGEST_TYPE_CPU),
               (unsigned long)energest_type_time(ENERGEST_TYPE_LPM),
               (unsigned long)energest_type_time(ENERGEST_TYPE_TRANSMIT),
               (unsigned long)energest_type_time(ENERGEST_TYPE_LISTEN));
      join_logged = 1;
    }
    etimer_reset(&check_timer);
  }

  PROCESS_END();
}