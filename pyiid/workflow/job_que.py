__author__ = 'christopher'
from time import sleep
while True:
    # if this doesn't blow up then there are simulations to run, run them!
    try:
        find_simulations(ran=False, skip=False)
        for sim in find_simulations(ran=False):
            run_simulation(sim)
            run_analysis(sim)
    # we blew up, implying that there were no more un-run simulations
    except StopIteration:
        # please hold while you enter more work for us to do!
        sleep(60)
