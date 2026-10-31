#ifndef TSCH_DEFINE_SYNC_H_
#define TSCH_DEFINE_SYNC_H_

#include "contiki.h"
#include <stdint.h>

uint8_t tsch_define_get_eb_channel_offset(uint64_t asn);
int tsch_define_should_send_eb(uint64_t asn);
void tsch_define_notify_minimal_cell_tx(int mac_tx_status);

#endif /* TSCH_DEFINE_SYNC_H_ */