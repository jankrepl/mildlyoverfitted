# The Lottery Ticket Hypothesis
## Installation
```bash
pip install -r requirements.txt
```

## Running experiments
The training logic is implemented inside of the script `main.py`. To
get more information about the CLI run

```bash
python main.py --help
```

If you want to run an entire grid search over different hyperparameters
you can use the `parallel_launch.sh` script. Note that it depends on a tool
called `parallel` ([more info](https://www.gnu.org/software/parallel/)). Note
that the script allows for dry runs (default behavior) and progress bars.

```bash
./parallel_launch.sh
```
