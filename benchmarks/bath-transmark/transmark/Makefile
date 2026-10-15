
CC       = gcc -std=gnu99
CFLAGS   = -g -O3 -fomit-frame-pointer -fstrict-aliasing -pthread
SIMDFLAGS= -msse2 -msse3
CPPFLAGS = 
LDFLAGS  = 
DEFS     = -DHAVE_CONFIG_H
LIBS     = -lhmmer -leasel    -lm
MPILIBS  = 
IMPLDIR  = impl_sse

SRCDIR   = ../src

ESLDIR   = ../../hmmer/easel
HMMERDIR = ../../hmmer

ESLINC   = -I${ESLDIR}
SRCINC   = -I${SRCDIR}

PROGS    = transmark-create\
           transmark-rocplot

PROGOBJS  =\
	transmark-create.o\
        transmark-rocplot.o

# beautification magic stolen from git 
QUIET_SUBDIR0 = +${MAKE} -C #space separator after -c
QUIET_SUBDIR1 = 
ifndef V
	QUIET_CC      = @echo '    ' CC $@;
	QUIET_GEN     = @echo '    ' GEN $@;
	QUIET_AR      = @echo '    ' AR $@;
	QUIET_SUBDIR0 = +@subdir=
	QUIET_SUBDIR1 = ; echo '    ' SUBDIR $$subdir; \
		        ${MAKE} -C $$subdir
endif

.PHONY: all dev distclean clean

all:    ${PROGS}
dev:    ${PROGS}

${PROGS}: %: %.o  ${HMMERDIR}/src/libhmmer.a ${ESLDIR}/libeasel.a 
	${QUIET_GEN}${CC} ${CFLAGS} ${DEFS} ${LDFLAGS} -L${HMMERDIR}/src -L${ESLDIR} -L${SRCDIR} -o $@ $@.o ${LIBS}

${PROGOBJS}: ${HMMERDIR}/src/hmmer.h ${HMMERDIR}/src/p7_config.h

.c.o: 
	${QUIET_CC}${CC} ${ESLINC} ${SRCINC} ${CFLAGS} ${SIMDFLAGS} ${DEFS} -o $@ -c $<		

clean:
	-rm -f *.o *~  ${PROGS} TAGS
	-rm -f *.gcno

################################################################
# transmark - Creates and runs DNA sequence benchmark with a 
# spefied rate of frameshfts for translated search using HMMER
# Version 1.0.0; May 2023
################################################################
