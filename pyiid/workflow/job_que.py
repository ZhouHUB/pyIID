__author__ = 'christopher'
from time import sleep
from simdb.search import find_simulation_document
from pyiid.workflow.simulation import run_simulation

i = 0
print 'Start job queue'
while True:
    print 'search for simulations to be run'
    sims = list(find_simulation_document(ran=False, skip=False,
                                         # error=False
                                         ))
    if len(sims) == 0:
        # we didn't find anything, implying that there were no more un-run simulations
        if i >= 30:
            print 'Idle for too long, exiting'
            exit()
        i += 1
        sleep(10)

    else:
        print 'Found {0} simulation enteries which have not been ran or flagged' \
              ' to be skipped'.format(len(sims))
        # run the simulations in the order they were added.
        for sim in reversed(sims):
            print 'start simulation number ', sim.id
            print 'simulation name', sim.name
            try:
                run_simulation(sim)
            except KeyboardInterrupt:
                print 'run ended'
                pass
            except:
                print 'Simulation number {} has errored'.format(sim.id)
                sim.error = True
                sim.skip = True
                sim.save()
                pass
