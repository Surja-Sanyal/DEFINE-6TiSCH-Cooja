CONTIKI_PROJECT = node
all: $(CONTIKI_PROJECT)

# Point to your contiki-ng-define folder in the workspace root
CONTIKI = ../../contiki-ng-define

MAKE_MAC = MAKE_MAC_TSCH
MAKE_ROUTING = MAKE_ROUTING_RPL_LITE

# Default parameters (can be overridden on the command line without wiping out CFLAGS)
SYNC_SCHEME ?= 1
SF_LENGTH ?= 101
NUM_CHANNELS ?= 16
EB_MULT ?= 1
ATP_SUBSLOTS ?= 2

CFLAGS += -DSYNC_SCHEME=$(SYNC_SCHEME)
CFLAGS += -DSF_LENGTH=$(SF_LENGTH)
CFLAGS += -DNUM_CHANNELS=$(NUM_CHANNELS)
CFLAGS += -DEB_MULT=$(EB_MULT)
CFLAGS += -DATP_SUBSLOTS=$(ATP_SUBSLOTS)

# Link math library for exp() and sqrt()
TARGET_LIBFILES += -lm

include $(CONTIKI)/Makefile.include