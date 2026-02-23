# PINN for Tumor Growth Modeling (Oncology)

## Overview
Physics-Informed Neural Network for modeling tumor volume progression using the logistic growth equation.

## Medical Relevance
Tumor growth prediction is critical in oncology for treatment planning and therapy response analysis.

## Governing Equation
dN/dt = rN(1 − N/K)

Where:
- N → Tumor cell population
- r → Growth rate
- K → Carrying capacity

## Method
The model learns tumor dynamics by minimizing:
- Data loss from observed tumor size
- Physics loss from logistic ODE residual

## Features
- Mechanistic tumor growth modeling
- Data-efficient learning
- Predictive tumor progression curve

## Run
python train.py