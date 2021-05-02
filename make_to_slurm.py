import os

from settings import *


OUTPUT_DIR = os.path.abspath(os.path.join(__file__, os.pardir, 'commands'))


def make_slurm_prefix(expeirment_name: str,
                      nodes: int,
                      hours: str,
                      output_file: str = 'output.out',
                      error_file: str = 'error.err'):
    return \
f"""#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J {expeirment_name}
## Liczba alokowanych węzłów
#SBATCH -N {nodes}
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time={hours}:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcontinualrl
## Specyfikacja partycji
#SBATCH -p plgrid
## Plik ze standardowym wyjściem
#SBATCH --output="{output_file}"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="{error_file}"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR
"""


def load_singularity_module():
    return "module add plgrid/tools/singularity"


commands = []
with open(PATH_TO_RUN, 'r') as file:
    for line in file:
        commands.append(line)


if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


wrapped_commands = []
for idx, command in enumerate(commands):
    run_file = make_slurm_prefix(EXPERIMENT_NAME, N_NODES, HOURS, f"output_{idx}.out", f"error_{idx}.err")
    run_file += "\n\n" + load_singularity_module() + '\n'
    run_file += "\n\n" + SINGULARITY_COMMAND + command + "\n"

    wrapped_command = os.path.join(OUTPUT_DIR, f"run_{idx}.sh")
    wrapped_commands.append(wrapped_command)

    with open(wrapped_command, 'w') as file:
        file.write(run_file)

with open(PATH_TO_SBATCH_COLLECTIVE, 'w') as file:
    file.writelines([f"sbatch {wrapped_command}\n" for wrapped_command in wrapped_commands])



