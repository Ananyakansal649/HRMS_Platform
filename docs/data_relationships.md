# Data Relationships Documentation

## Enterprise HR AI — Day 1: Data Foundation

This document records the verified relationships between the five raw datasets used in this project.

---

## Dataset Overview

| Dataset | Rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| employee_attrition.csv | 1,470 | 35 | Attrition prediction |
| hr_performance_engagement.csv | 2,845 | 28 | Engagement/performance analytics |
| occupation_data.csv | 1,016 | 3 | Occupation/role master |
| essential_skills.csv | 18,200 | 15 | Essential skills by role |
| software_skills.csv | 31,821 | 7 | Software/technology skills by role |

---

## Verified Relationships

### 1. employee_attrition ↔ hr_performance_engagement

| Property | Value |
| --- | --- |
| **Join Key** | `EmployeeNumber` ↔ `Employee ID` |
| **Relationship** | One-to-one (both IDs are unique) |
| **Overlap** | 731 out of 1,470 attrition IDs (49.7%) |
| **Evidence** | Both columns are unique in their respective datasets. 731 employee IDs appear in both. |
| **Decision** | Partially joinable. Only ~50% of attrition employees have engagement records. This is expected — not all employees may have completed engagement surveys. |

**Note:** The engagement dataset has 2,845 rows vs 1,470 in attrition. The engagement dataset appears to contain historical records (multiple survey periods per employee), while attrition is a snapshot of current employees. The 731 overlap represents employees who appear in both datasets.

### 2. employee_attrition ↔ occupation_data

| Property | Value |
| --- | --- |
| **Join Key** | `JobRole` ↔ `Title` |
| **Relationship** | Many-to-one |
| **Overlap** | 0 direct matches on exact string |
| **Evidence** | Attrition uses simplified role names (e.g., "Sales Executive"), occupation_data uses O*NET titles (e.g., "Sales Managers"). |
| **Decision** | Requires mapping table. The JobRole values in attrition are company-specific role names that need to be mapped to O*NET occupation titles. This mapping should be built during Day 3. |

**Attrition JobRole values:** Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources

### 3. occupation_data ↔ essential_skills

| Property | Value |
| --- | --- |
| **Join Key** | `O*NET-SOC Code` |
| **Relationship** | One-to-many |
| **Overlap** | 910 out of 1,016 occupation codes (89.6%) |
| **Evidence** | All 910 essential_skills codes exist in occupation_data (100% overlap from skills side). |
| **Decision** | Confirmed joinable on O*NET-SOC Code. One occupation has many skill entries (importance + level per skill). |

### 4. occupation_data ↔ software_skills

| Property | Value |
| --- | --- |
| **Join Key** | `O*NET-SOC Code` |
| **Relationship** | One-to-many |
| **Overlap** | 923 out of 1,016 occupation codes (90.8%) |
| **Evidence** | All 923 software_skills codes exist in occupation_data (100% overlap from skills side). |
| **Decision** | Confirmed joinable on O*NET-SOC Code. One occupation has many software/technology tool entries. |

---

## Relationship Diagram (Verified)

```
EMPLOYEE (attrition)
   |
   +-- EmployeeNumber ↔ Employee ID --> Engagement Data
   |                                    (partial overlap: 49.7%)
   |
   +-- JobRole (company-specific) --> [MAPPING NEEDED] --> Occupation Data (O*NET)
                                                              |
                                                              +-- O*NET-SOC Code --> Essential Skills
                                                              |
                                                              +-- O*NET-SOC Code --> Software Skills
```

---

## Employee-Level Current Skills

**Finding:** The five raw datasets do **NOT** contain per-employee current skills.

- `essential_skills.csv` contains skill requirements at the **occupation/role level** (not per employee)
- `software_skills.csv` contains software requirements at the **occupation/role level** (not per employee)
- Neither `employee_attrition.csv` nor `hr_performance_engagement.csv` contain skill-related columns

**Impact:** A real skill-gap calculation requires knowing what skills each employee currently has. This data must be sourced or constructed on Day 3 (per the professor's plan: `notebooks/12_employee_skills.ipynb`).

---

## Unresolved Issues for Day 3

1. **JobRole ↔ O*NET Title mapping:** Need to build a mapping from company-specific JobRole names to O*NET occupation titles
2. **Employee skills table:** Must be built (synthetic or from external source) since the raw datasets don't contain per-employee skill data
3. **Engagement dataset structure:** Contains historical survey data with multiple records per employee — need to decide how to aggregate for analysis

---

*Generated on Day 1 of the Enterprise HR AI project.*
