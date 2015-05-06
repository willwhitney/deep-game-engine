import os
import sys

dry_run = '--dry-run' in sys.argv
local   = '--local' in sys.argv
detach  = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

networks_prefix = "networks/"
networks_suffix = ".t7"

jobs = [
        # test job
        # {
        #     'name': 'test',
        #     'import': 'DQN_saving_params',
        #     'rom': 'breakout',
        #     'learn': False,
        #     'steps': '1000',
        # },

        # real jobs
        {
            'name': 'DQN_asteroids',
            'import': None,
            'rom': 'asteroids',
            'learn': True,
        },
        # {
        #     'name': 'DQN_fishing_derby_trained',
        #     'import': 'DQN_fishing_derby_saving_params',
        #     'rom': 'fishing_derby',
        #     'learn': False,
        #     'steps': '10000000',
        # },
        # {
        #     'name': 'DQN_freeway_trained',
        #     'import': 'DQN_freeway_saving_params',
        #     'rom': 'freeway',
        #     'learn': False,
        #     'steps': '10000000',
        # },
        # {
        #     'name': 'DQN_seaquest_trained',
        #     'import': 'DQN_seaquest_saving_params',
        #     'rom': 'seaquest',
        #     'learn': False,
        #     'steps': '10000000',
        # },
        # {
        #     'name': 'DQN_space_invaders_trained',
        #     'import': 'DQN_space_invaders_saving_params',
        #     'rom': 'space_invaders',
        #     'learn': False,
        #     'steps': '10000000',
        # },
    ]

if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    assert 'name' in job
    assert 'import' in job
    assert 'rom' in job
    assert 'learn' in job
    assert 'steps' in job

    name = job.pop('name', None)
    import_file = job.pop('import', None)
    rom = job.pop('rom', None)
    learn = job.pop('learn', None)
    steps = job.pop('steps', None)

    assert job == {}

    # import_string = ''
    if import_file != None:
        import_string = "-network '" + networks_prefix + import_file + networks_suffix + "'"
    else:
        import_string = ""

    jobcommand = "th train_agent.lua $args -name '{name}' {import_string} -steps {steps} {learn}".format(
        name = name,
        import_string = import_string,
        steps = steps,
        learn = '-learn' if learn else '')

    script_path = 'run_gpu_' + name
    os.system('cp run_gpu_template ' + script_path)

    with open(script_path, 'a') as script_file:
        script_file.write('\n')
        script_file.write(jobcommand)
        script_file.write('\n')


    print(jobcommand + " " + rom)
    if local:
        if not dry_run:
            if detach:
                os.system('bash ' + script_path + ' ' + rom + ' 2> slurm_logs/' + name + '.err 1> slurm_logs/' + name + '.out &')
            else:
                os.system('bash ' + script_path + ' ' + rom)

    else:
        with open('slurm_scripts/' + name + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name"+"=" + name + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" + name + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + name + ".err\n")
            slurmfile.write('bash ' + script_path + ' ' + rom)

        if not dry_run:
            os.system("sbatch -N 1 -c 1 --gres=gpu:1 -p gpu --mem=16000 --time=6-23:00:00 slurm_scripts/" + name + ".slurm &")
