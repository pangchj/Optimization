# Optimization codes

Here are some optimization codes that I wrote that may be helpful when you want to learn or implement an optimization algorithm. I take the code description from textbooks and write python codes so that the calculations are done step by step to match the  numerical results given there.

I have implemented:
- Cutting Stock problem (based on example in Chvatal's "Linear Programming")
- Dantzig Wolfe decomposition (based on example in Chvatal's "Linear Programming")
- Benders decomposition (based on example in Lasdon's "Optimization Theory for Large Systems")

To run the codes, make sure you have PULP installed, and then run with a python command line like ```python Benders.py``` for example. You can then compare the outputs to those given in the references. 

If you are looking for free serious optimization software in Python, you can try these first:
- CVX: https://cvxopt.org/index.html
- Google OR-Tools: https://developers.google.com/optimization
- PULP: https://coin-or.github.io/pulp/index.html

I have a background in nonlinear programming, but I haven't got interested in writing such codes yet. (It is unlikely that one would need to program the conjugate gradient method in practice, while some of the modern first order methods are too easy to program.) I might get something later. 

Any feedback is appreciated! If you see any issue or any way to improve this repository, do let me know. (If I don't have time to fix the issue, at least I can reflect that there is an issue.) 
