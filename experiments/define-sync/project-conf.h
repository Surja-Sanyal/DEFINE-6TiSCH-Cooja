#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

/* Enable TSCH and RPL-Lite */
#define MAKE_MAC MAKE_MAC_TSCH
#define MAKE_ROUTING MAKE_ROUTING_RPL_LITE

/* Parameterized via Makefile CFLAGS */
#ifndef SYNC_SCHEME
#define SYNC_SCHEME 1 /* 1:DEFINE, 2:MC, 3:EDBA, 4:ECV, 5:ATP, 6:TRGB, 7:CSMJ, 8:EASE */
#endif

#ifndef SF_LENGTH
#define SF_LENGTH 101
#endif

#ifndef NUM_CHANNELS
#define NUM_CHANNELS 16
#endif

#ifndef EB_MULT
#define EB_MULT 1
#endif

#ifndef ATP_SUBSLOTS
#define ATP_SUBSLOTS 2
#endif

#define TSCH_SCHEDULE_CONF_DEFAULT_LENGTH SF_LENGTH
#define TSCH_CONF_DEFAULT_TIMESLOT_LENGTH 10000 /* 10 ms */
#define ENERGEST_CONF_ON 1

/* Custom hooks into Contiki-NG TSCH */
#define TSCH_CALLBACK_NEW_TIME_SOURCE define_on_new_time_source
#define TSCH_CONF_EB_PERIOD (EB_MULT * SF_LENGTH * (CLOCK_SECOND / 100))

/* Ensure Enhanced Beacons advertise the custom slotframe schedule */
#define TSCH_PACKET_CONF_EB_WITH_SLOTFRAME_AND_LINK 1

/* Channel hopping sequence length adjustment (4, 8, 12, 16) */
#if NUM_CHANNELS == 4
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE TSCH_HOPPING_SEQUENCE_4_4
#elif NUM_CHANNELS == 8
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17, 22, 19 }
#elif NUM_CHANNELS == 12
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17, 22, 19, 11, 14, 18, 23 }
#else
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE TSCH_HOPPING_SEQUENCE_16_16
#endif

/* Optimal pledge channel scan duration: SF_LENGTH * 10 ms per channel */
#define TSCH_CONF_CHANNEL_SCAN_DURATION (SF_LENGTH * (CLOCK_SECOND / 100))

/* Z1 Mote RAM Optimization to prevent stack overflow at $00e2e */
#define NBR_TABLE_CONF_MAX_NEIGHBORS 8
#define NETSTACK_MAX_ROUTE_ENTRIES 8
#define UIP_CONF_BUFFER_SIZE 240
#define UIP_CONF_UDP 1
#define UIP_CONF_TCP 0
#define SICSLOWPAN_CONF_FRAG 0
#define TSCH_LOG_CONF_PER_SLOT 0

/* Include node.h so tsch.c sees the define_on_new_time_source() prototype */
#include "node.h"

#endif /* PROJECT_CONF_H_ */