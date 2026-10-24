"""Training infrastructure: what the runs ran on, measured while they run, reported for the paper.

    sysinfo   one record per job: GPUs (incl. MIG slices), driver, CUDA, CPU, memory, Slurm, software versions
    monitor   samples GPUs, CPU, memory and training progress every second (dashboard: infra.html)
    report    compute per method (GPU-hours), hardware table, timelines: PDF pages, PNG + PDF figures, LaTeX table

Files, next to the experiments of one results folder (PAPER_A_OUT):
    infra/<job id>/system.json        sysinfo record of that job
    infra/<job id>/timeseries.csv     one row per second: GPU memory / power / temperature, CPU, RAM, throughput
    dashboard/infra/live.json         the latest samples, polled by the dashboard's infrastructure page
    infra/progress/<run>.json         each training process's progress (written by train_one)
    <EXP>/infrastructure.{json,md,tex} and <EXP>/figures/infra_*.{png,pdf}   the summary of an experiment
"""
