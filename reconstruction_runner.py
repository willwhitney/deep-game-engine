import os
import sys

dry_run = '--dry-run' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


# networks_dir = '/om/user/wwhitney/facegen_networks/'
base_networks = {
    }


# Don't give it a save name - that gets generated for you
jobs = [

        # A couple of quick tests
        {
            'datasetdir': 'dataset-trained-more',
            'num_train_batches': 24000,
            'num_test_batches': 2800,
            'epoch_size': 5,
            'tests_per_epoch': 5,
            'learning_rate': '-0.00001'
        },
        {
            'datasetdir': 'dataset-trained-more',
            'num_train_batches': 24000,
            'num_test_batches': 2800,
            'grayscale': True,
            'epoch_size': 5,
            'tests_per_epoch': 5,
            'learning_rate': '-0.0001'
        },


        # the real jobs
        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'learning_rate': '-0.0001'
        # },
        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'learning_rate': '-0.00005',
        # },
        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'learning_rate': '-0.00001'
        # },

        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'dim_hidden': 400,
        #     'feature_maps': 128,
        #     'learning_rate': '-0.0001'
        # },
        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'dim_hidden': 400,
        #     'feature_maps': 128,
        #     'learning_rate': '-0.00001'
        # },

        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'dim_hidden': 40,
        #     'feature_maps': 48,
        #     'learning_rate': '-0.0001'
        # },
        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'dim_hidden': 40,
        #     'feature_maps': 48,
        #     'learning_rate': '-0.00001'
        # },

        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'grayscale': True,
        #     'learning_rate': '-0.0001'
        # },
        # {
        #     'datasetdir': 'dataset-trained-more',
        #     'num_train_batches': 24000,
        #     'num_test_batches': 2800,
        #     'grayscale': True,
        #     'learning_rate': '-0.00001'
        # }
    ]

if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    jobname = "reconstruction"
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " --" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                flagstring = flagstring + " --" + flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " --" + flag + " " + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = "th reconstruction_main.lua" + flagstring
    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write(jobcommand)

    # if not os.path.exists(jobname):
    #     os.makedirs(jobname)

    # with open(jobname + '/generating_parameters.txt', 'w') as paramfile:
    #     paramfile.write(str(job))

    print(jobcommand)
    if not dry_run:
        os.system("sbatch -N 1 -c 2 --gres=gpu:1 -p gpu --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")




