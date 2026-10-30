#ifndef NODE_H_
#define NODE_H_

#include "contiki.h"

/* Forward declaration of TSCH neighbor struct for the callback prototype */
struct tsch_neighbor;

/* Callback invoked by Contiki-NG TSCH when a node selects or switches its time source */
void define_on_new_time_source(const struct tsch_neighbor *old, const struct tsch_neighbor *new);

#endif /* NODE_H_ */