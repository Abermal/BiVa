# Bias Variance Agent

Biva library serves as tool to handle bias-variance tradeoff while solving supervised learning problems.
Agent provides API-independent approach for model exploration, evaluation and hyperparameter optimisation.

Read the documentation at [dsrg.fh-zwickau.de](https://doc.dsrg.fh-zwickau.de/bias-variance-agent).  
Biva is compatible with: **Python 3.8**.


## Getting Started

To get started with the sources clone [this](https://dsrg.fh-zwickau.de/research/bias-variance-agent.git) git repository 
to any *biva_folder* with

 `git clone https://dsrg.fh-zwickau.de/research/bias-variance-agent.git` 
 
 and follow these steps:
* with `pip` and `venv`:
    1. open a pip terminal in *biva_folder*. You can make sure that pip is up-to-date by running:\
    `python -m pip install --upgrade pip`\
virtualenv is used to manage Python packages for different projects. Install it with:\
    `python -m pip install --user virtualenv`
    2. Create a virtual environment. \
    `python -m venv .env`
    3. Activate the virtual environment. \
    `.\.env\Scripts\activate`
    3. Run `pip install -e .` to install this library in development mode
* with `conda`:
    1. Open `conda` terminal in *biva_folder*.
    2. Create a conda environment in `.env` folder. \
    `conda env create --force --prefix ./.env python=3.8.*`
    3. Activate the conda environment. \
    `conda activate ./.env` 
    4. Run `pip install -e .` to install this library in development mode


## Contribution guidelines
To contribute to the project please use pull a request.
Before sending it do the follwing:
- Run tests using `pytest` to make sure everything is working.
- Use pylint to ensure code quallity.


## License
Currently undefined.


## Scientific Supervisor
- Mike Espig (mike.espig@fh-zwickau.de)


## Authors
- Marcel Becker (marcel.becker@fh-zwickau.de)
- Kostiantyn Pysanyi (kostiantyn.pysanyi.jud@fh-zwickau.de)


## Acknowledgments
The research was funded by:  
[*Federal Ministry of Education and Research of Germany (BMBF)*](https://www.bmbf.de/) and  
[*Saxony Ministry of Science and Art (SMWK)*](http://www.smwk.sachsen.de/).
