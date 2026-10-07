## Create a simple JEPA Architecture

## First run




# Notes:
Q1: Inter graph and intra graph relationships

-> Input: Evidence, Given: result, Output: Claim

Gen.function(Claim/Evidence, Result) 



Evidence:  [+5]*512 - (1, 512)
Result: Sine(2*pi*0.1t) - (1, 512)
-----
Claim:  Sine(2*pi*0.5t) - (1, 512)

Input: [BS, 2, 512]
Output: [BS, 512]

## 22.09.2026
* Lokesh and Sasi disagree on the subencoder implementation.
 - We need to revisit whether we do require subencoder for sure or does it make sense to add it than training with a single graph context encoder in JEPA.