# System Architecture

## Model
Fully connected neural network:
Input: time (t)  
Output: Tumor size N(t)

## Loss Function
Total Loss = Data Loss + Physics Loss

### Physics Loss
Residual of logistic growth equation:
dN/dt − rN(1 − N/K)

### Data Loss
MSE between predicted and observed tumor size.

## Training Workflow
1. Sample time collocation points
2. Compute tumor prediction
3. Compute derivative via automatic differentiation
4. Enforce logistic growth constraint
5. Optimize combined loss

## Output
Tumor growth curve visualization