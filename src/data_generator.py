import os
import numpy as np
import pandas as pd


def _performance_label(score: float) -> str:
    if score >= 75:
        return "High Performer"
    if score >= 55:
        return "Average Performer"
    return "Needs Improvement"


def generate_employee_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic HR dataset for analytics and ML."""
    rng = np.random.default_rng(random_state)

    departments = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations", "Support"]
    roles = {
        "Engineering": ["Software Engineer", "Data Analyst", "QA Engineer", "Team Lead"],
        "Sales": ["Sales Executive", "Business Associate", "Account Manager"],
        "HR": ["HR Executive", "Recruiter", "HR Manager"],
        "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
        "Marketing": ["SEO Analyst", "Content Strategist", "Marketing Manager"],
        "Operations": ["Operations Associate", "Process Analyst", "Operations Manager"],
        "Support": ["Customer Support", "Service Associate", "Support Lead"],
    }
    educations = ["Graduate", "Postgraduate", "MBA", "Diploma"]
    genders = ["Female", "Male", "Other"]

    rows = []
    for i in range(1, n_samples + 1):
        dept = rng.choice(departments, p=[0.24, 0.16, 0.10, 0.10, 0.12, 0.16, 0.12])
        role = rng.choice(roles[dept])
        age = int(rng.integers(22, 58))
        experience = int(np.clip(age - rng.integers(20, 28), 0, 35))
        education = rng.choice(educations, p=[0.46, 0.30, 0.14, 0.10])
        gender = rng.choice(genders, p=[0.45, 0.52, 0.03])

        base_salary = {
            "Engineering": 75000,
            "Sales": 62000,
            "HR": 58000,
            "Finance": 70000,
            "Marketing": 65000,
            "Operations": 60000,
            "Support": 50000,
        }[dept]
        monthly_salary = int(base_salary + experience * 2500 + rng.normal(0, 12000))
        monthly_salary = int(np.clip(monthly_salary, 25000, 180000))

        training_hours = int(np.clip(rng.normal(34, 12), 5, 80))
        attendance_percent = float(np.clip(rng.normal(90, 6), 65, 100))
        projects_completed = int(np.clip(rng.normal(6 + experience * 0.15, 2.5), 0, 18))
        overtime_hours = int(np.clip(rng.normal(14, 9), 0, 45))
        satisfaction_score = int(np.clip(rng.normal(7.0, 1.8), 1, 10))
        work_life_balance = int(np.clip(rng.normal(6.8, 1.7), 1, 10))
        manager_rating = float(np.clip(rng.normal(3.5, 0.7), 1, 5))
        last_promotion_years = int(np.clip(rng.normal(3.0, 1.8), 0, 10))
        remote_days_per_month = int(np.clip(rng.normal(6, 4), 0, 20))
        skill_score = int(np.clip(rng.normal(68 + experience * 0.6, 12), 35, 98))

        # Department effects
        dept_bonus = {
            "Engineering": 4,
            "Sales": 2,
            "HR": 1,
            "Finance": 3,
            "Marketing": 2,
            "Operations": 1,
            "Support": 0,
        }[dept]

        # Synthetic performance score formula
        perf_score = (
            0.22 * attendance_percent
            + 1.4 * projects_completed
            + 0.18 * training_hours
            + 2.6 * satisfaction_score
            + 3.8 * manager_rating
            + 0.20 * skill_score
            + 0.45 * experience
            + dept_bonus
            - 0.55 * overtime_hours
            - 0.75 * max(last_promotion_years - 4, 0)
            - 0.8 * max(5 - work_life_balance, 0)
            + rng.normal(0, 5)
        )
        perf_score = float(np.clip(perf_score, 30, 98))
        performance_label = _performance_label(perf_score)

        attrition_risk = (
            0.35 * (10 - satisfaction_score)
            + 0.28 * overtime_hours
            + 0.12 * max(last_promotion_years - 3, 0)
            + rng.normal(0, 2)
        )
        attrition_risk = float(np.clip(attrition_risk, 5, 95))

        rows.append(
            {
                "employee_id": f"EMP{i:04d}",
                "age": age,
                "gender": gender,
                "department": dept,
                "job_role": role,
                "education": education,
                "years_experience": experience,
                "monthly_salary": monthly_salary,
                "training_hours": training_hours,
                "attendance_percent": round(attendance_percent, 1),
                "projects_completed": projects_completed,
                "overtime_hours": overtime_hours,
                "satisfaction_score": satisfaction_score,
                "work_life_balance": work_life_balance,
                "manager_rating": round(manager_rating, 1),
                "last_promotion_years": last_promotion_years,
                "remote_days_per_month": remote_days_per_month,
                "skill_score": skill_score,
                "performance_score": round(perf_score, 1),
                "performance_label": performance_label,
                "attrition_risk": round(attrition_risk, 1),
            }
        )

    df = pd.DataFrame(rows)

    # Add light missingness for realism
    for col in ["training_hours", "satisfaction_score", "manager_rating"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    return df


def save_dataset(filepath: str, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_employee_dataset(n_samples=n_samples, random_state=random_state)
    df.to_csv(filepath, index=False)
    return df
