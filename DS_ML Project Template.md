DS/ML Project Template

> This template can be used to create user stories for ML/DS projects; it does not include productionalizing.

---

# User Stories and Considerations

- **Create Project Plan** - `5 pts`
    - This is probably a day of actual work, maybe longer; but most certainly spread out over many days, especially if interacting with stakeholders.
    - Define: 
        - Business Goal
            - Problem: What is the pain/point we are trying to solve?
            - Solution: How will the output of the project attempt to solve that problem? How will the output be used?
              - Should the solution be ML/DS or are we over-complicating the solution?
            - Economics: What is the cost of the pain point? What is the value of the solution? What is the cost to create the solution? 
                - e.g. in classification model, what is the actual or relative value of a TP/FP/FN/TN
        - Customer Acceptance 
            - Define Customer/Stakeholders 
            - Definition of Done
            - Definition of Success and Success Metrics
        - Differentiate between MVP, V1, Long-term
    - Data
      - Brainstorm ideal data availability needed to solve this problem
      - Identify Existing/Possible Data Sources
    - Define project risks.
- **Data Acquisition, Cleaning, Exploring**
    - Ingest/Clean/Validate Data - `8 pts`
        - Set up reproducible data pipeline; could be a simple script or makefile
            - (e.g. if we need to update the data-sources, we don’t want to run 5 different notebooks manually)
        - Cross-reference ideal data with actual data
        - Document current data, limitations, cross reference ideal data with actual data.
        - Document possible future improvements on data cleaning or additional data to look at.
    - EDA - `5 pts` 
        - Descriptive analysis of data (descriptive stats, summaries, profiles, trends, correlations) 
- **DS / ML**
    - Feature engineering - `5 pts`
        - Take the data from the `Data Acquisition, Cleaning, and Exploring` task and create the dataset you need to answer business question (e.g. model).
        - Unit Testing
    - Model Building and Evaluation - `8 pts`
        - Iterative process of trying new things and seeing what works and does not (transformations/PCA, etc.)
    - Report on model performance - `5 pts`
        - Is the performance acceptable for business needs (i.e. does model provide positive economic value)
- **Final Report** - `5 pts`
    - Update project plan with executive summary and/or create final report to customer.
      - Acknowledge Customer Acceptance Criteria
      - Document Next Steps

> It would be nice if we had steps in the process to ensure we are getting feedback from group as we are doing the project (rather than afterwards when it is too late to incorporate feedback).

---

# Summary

- Document Project Overview - `5`
- Ingest/Clean Data - `8`
- Exploratory Data Analysis - `5`
- ===
- Feature Engineering - `5`
- Model Building/Evaluation - `8`
- Model Report -  `5`
- Reviews (Project/Code) - `?`
- Final Report - `5`

`5 + 8 + 5 + 5 + 8 + 5 + 5 = 41 pts`

---

# User Story Points to Level-of-Effort
```

| Points | Approximate Level of Effort |
| —————— |:———————————————————————————:|
| 1      | 1 hour                      |
| 2      | 2 hours                     |
| 3      | 1/2 day                     |
| 5      | 1 day                       |
| 8      | 2 days                      |
| 13     | 3+ days                     |
```

---

# References
[What is the Team Data Science Process? - Azure Architecture Center | Microsoft Docs](https://docs.microsoft.com/en-us/azure/architecture/data-science-process/overview)

---