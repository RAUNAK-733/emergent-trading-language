param(
    [int[]]$Seeds = @(7, 17, 42),
    [int]$Updates = 25000
)

$ErrorActionPreference = "Stop"
$RunDirectories = @()

foreach ($Seed in $Seeds) {
    $RunDirectory = "runs/seed-$Seed"
    $FigureDirectory = "figures/seed-$Seed"
    $RunDirectories += $RunDirectory

    python main.py train --fresh --seed $Seed --updates $Updates --run-dir $RunDirectory
    python main.py analyze --run-dir $RunDirectory --figure-dir $FigureDirectory
}

python main.py compare --runs $RunDirectories
