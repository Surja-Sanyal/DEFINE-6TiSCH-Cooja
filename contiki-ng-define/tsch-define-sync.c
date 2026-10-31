#include "contiki.h"
#include "net/mac/tsch/tsch.h"
#include "net/routing/rpl-lite/rpl.h"
#include "sys/node-id.h"
#include "lib/random.h"
#include <math.h>

/* Synchronization Mechanism Selector:
 * 'P' = DEFINE (Proposed), 'E' = 6TiSCH-MC, 'R' = EDBA-SA,
 * 'D' = ECV-SA, 'A' = ATP, 'T' = TRGB, 'C' = Chang2026, 'S' = EASE
 */
#ifndef SYNC_SCHEME
#define SYNC_SCHEME 'P'
#endif

#ifndef SYNC_NUM_CHANNELS
#define SYNC_NUM_CHANNELS 16
#endif

#ifndef SYNC_SLOTFRAME_SIZE
#define SYNC_SLOTFRAME_SIZE 101
#endif

static uint8_t chang_be = 1; /* Backoff exponent for Chang 2026 ('C') */

static uint16_t
get_node_dodag_height(void)
{
  if(tsch_is_coordinator) {
    return 0;
  }
  if(!curr_instance.used || curr_instance.dag.rank == RPL_INFINITE_RANK) {
    return 1;
  }
  return (uint16_t)(curr_instance.dag.rank / RPL_MIN_HOP_RANK_INC);
}

static uint16_t
get_local_density(void)
{
  return (uint16_t)rpl_neighbor_count();
}

uint8_t
tsch_define_get_eb_channel_offset(uint64_t asn)
{
  uint32_t sf_num = (uint32_t)(asn / SYNC_SLOTFRAME_SIZE);
  uint16_t h = get_node_dodag_height();

  switch(SYNC_SCHEME) {
    case 'P': /* DEFINE (ASSIGN): (ASN + h) mod C */
      return (uint8_t)(h % SYNC_NUM_CHANNELS);

    case 'E': /* 6TiSCH-MC: channelOffset = 0 */
    case 'C': /* Chang 2026: minimal cell channelOffset = 0 */
      return 0;

    case 'R': /* EDBA-SA: uniform random channel offset */
      return (uint8_t)(random_rand() % SYNC_NUM_CHANNELS);

    case 'D': /* ECV-SA: deterministic offset based on node ID */
      return (uint8_t)(node_id % SYNC_NUM_CHANNELS);

    case 'T': { /* TRGB: Time-Variant Red-Green-Blue sub-band offset */
      uint8_t rgb_phase = (uint8_t)(sf_num % 3);
      uint8_t sub_band = (uint8_t)(SYNC_NUM_CHANNELS / 3);
      if(sub_band == 0) sub_band = 1;
      return (uint8_t)((rgb_phase * sub_band + (node_id % sub_band)) % SYNC_NUM_CHANNELS);
    }

    case 'S': { /* EASE: Autonomous hash of node_id */
      uint32_t hash = ((uint32_t)node_id * 2654435761UL) % SYNC_NUM_CHANNELS;
      return (uint8_t)hash;
    }

    default:
      return 0;
  }
}

int
tsch_define_should_send_eb(uint64_t asn)
{
  uint32_t sf_num = (uint32_t)(asn / SYNC_SLOTFRAME_SIZE);
  uint16_t nbr = get_local_density();
  uint16_t rand_val = random_rand() % 1000;

  switch(SYNC_SCHEME) {
    case 'P': { /* DEFINE (ADAPT): exp(-(ASN * sqrt(1 + nbr)) / (C * S)) */
      double exponent = -1.0 * ((double)sf_num * sqrt(1.0 + (double)nbr)) / (double)SYNC_NUM_CHANNELS;
      uint16_t prob_permille = (uint16_t)(exp(exponent) * 1000.0);
      if(prob_permille < 10) {
        prob_permille = 10;
      }
      return rand_val < prob_permille;
    }

    case 'E': /* 6TiSCH-MC: P_eb = 0.1 */
    case 'A':
      return rand_val < 100;

    case 'R': /* EDBA-SA & ECV-SA: Deterministic once per interval */
    case 'D':
      return 1;

    case 'T': { /* TRGB: Red (0.8), Green (0.4), Blue (0.1) */
      uint8_t phase = (uint8_t)(sf_num % 3);
      uint16_t p_trgb = (phase == 0) ? 800 : ((phase == 1) ? 400 : 100);
      return rand_val < p_trgb;
    }

    case 'C': { /* Chang 2026: Shared-cell backoff 2^(-BE) */
      uint16_t p_chang = 1000 >> chang_be;
      return rand_val < p_chang;
    }

    case 'S': { /* EASE: Energy-aware scaling */
      uint16_t p_ease = (uint16_t)(100.0 / (1.0 + 0.5 * (double)nbr));
      if(p_ease < 20) p_ease = 20;
      return rand_val < p_ease;
    }

    default:
      return rand_val < 100;
  }
}

void
tsch_define_notify_minimal_cell_tx(int mac_tx_status)
{
  if(SYNC_SCHEME == 'C') {
    if(mac_tx_status == MAC_TX_COLLISION || mac_tx_status == MAC_TX_NOACK) {
      if(chang_be < 5) chang_be++;
    } else if(mac_tx_status == MAC_TX_OK) {
      if(chang_be > 1) chang_be--;
    }
  }
}