# Clarify: Improving Model Robustness With Natural Language Corrections

This repository contains code for the non-expert user interface in the [Clarify paper](https://arxiv.org/abs/2402.03715).

## Instructions

First, install the Python dependencies:

```bash
pip install -r requirements.txt
```

The interface is a Flask app that can be run locally.
To run the server, run either:

Download the features first:

```bash
bash download_features.sh
```

Then, run the server with either of the following commands:

```bash
# For local debugging
python server.py

# For deployment
gunicorn -w 4 server:app --bind 0.0.0.0:5002
```

Directly running the server script is for local debugging, and gunicorn is for deployment for handling multiple users at once.
For either option, the server should be accessible through <http://localhost:5002>.

All logs are saved in `logs/`.

## User Study

We ran our user study through [Prolific](https://www.prolific.com/).
We did minimal filtering to ensure that participants were fluent in English and familiar with similar interfaces. Our filters were:

- Located in the USA
- Primary language is English
- 95% or higher approval rate on Prolific
- Over 50 submissions on Prolific
- Some computer programming experience

Our full study description shown to participants is [here](study_description.md).

## Citation

If our work is useful for your own, you can cite our paper using the following BibTeX entry:

```bibtex
@article{lee2024clarify,
  title={Clarify: Improving Model Robustness With Natural Language Corrections},
  author={Lee, Yoonho and Lam, Michelle S and Vasconcelos, Helena and Bernstein, Michael S and Finn, Chelsea},
  journal={arXiv preprint arXiv:2402.03715},
  year={2024}
}
```
