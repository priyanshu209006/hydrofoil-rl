# Hydrofoil shape optimization with reinforcement learning

Most shape optimization work looks like this: generate thousands of random shapes, score them all with a solver, keep the best ones (basically a genetic algorithm). It works, but it burns a huge number of solver calls.

This project does something different. It treats shape design as a step by step game:

- the agent starts from some random hydrofoil section
- it makes 40 small moves, and each move changes one CST (Kulfan) weight a little bit
- after every move, NeuralFoil (a fast neural network version of XFOIL) returns the new L/D in about a millisecond
- a PPO agent (Stable-Baselines3) learns which moves actually help

The result is not just one optimized shape. Because every training episode randomizes the starting shape, the Reynolds number and the angle of attack, the trained policy is a reusable skill: point it at any foil at any condition and it sculpts an improvement in 40 moves.

## How the problem is set up

| piece | what it is |
|---|---|
| state | 8 upper + 8 lower CST weights, current L/D, mid chord thickness, log Re, alpha |
| action | 32 choices: pick one of the 16 weights, move it up or down |
| reward | change in L/D for that move, infeasible moves get rejected with a small penalty |
| episode | 40 moves on a random task |

Starting shapes come from real airfoil families (Selig, Eppler, NASA GA, Clark, MH, wind turbine sections) plus random NACA 4 digit sections, all fitted into the same 8 + 8 CST basis.

Shape realism is enforced with a few simple rules taken from real sections: thickness between 4% (mid chord) and 18% (section max), camber under 6%, no razor thin pinch, no wavy surfaces. A nice CST property does the heavy lifting: with all weights >= 0 the surface cannot cross itself, so the agent physically cannot produce an invalid shape.

## Results

Trained for 350k steps (about 9 minutes on a laptop CPU), then tested on 8 tasks that were never seen in training:

- mean L/D change: PPO +15.4, random walk -1.2
- the policy improved the L/D on 8 out of 8 tasks
- zero rejected moves during evaluation, so all shapes stayed inside the design rules
- final shapes were re-checked with a bigger NeuralFoil model through a different code path, and the improvements held up

## Files

- `hydrofoil_rl.ipynb` : everything (geometry, environment, training, evaluation), already executed with outputs
- `requirements.txt` : python dependencies

## How to run

```
pip install -r requirements.txt
jupyter lab   # open the notebook and run all cells
```

Training takes around 9 minutes on CPU. There is no GPU requirement on purpose: the policy network is tiny, so a GPU would spend more time on overhead than on math, and the actual bottleneck is the environment loop plus NeuralFoil which run on CPU anyway.

## Notes

- The solver here is a surrogate. For anything serious, export the CST weights of the shapes you like and check them in XFOIL or a RANS solver.
- Useful references: Kulfan, "CST Universal Parametric Geometry Representation Method" (AIAA Journal 2007) and Ceze, Hayashi, "A Study of the CST Parameterization Characteristics" (AIAA 2009).
