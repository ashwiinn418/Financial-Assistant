import json
import random
from typing import Dict, List, Tuple

class RiskProfileGenerator:
    def __init__(self):
        self.questions = [
            ("What is your primary source of income?", ["Salary", "Business", "Investments", "Others"]),
            ("How stable is your income?", ["Very stable", "Somewhat stable", "Unstable"]),
            ("What percentage of your income do you save or invest each month?", ["Less than 10%", "10-30%", "More than 30%"]),
            ("Do you have any outstanding loans or EMIs?", ["Yes, multiple loans", "Yes, but manageable", "No"]),
            ("What percentage of your monthly income goes towards loan or credit card payments?", ["Less than 20%", "20-50%", "More than 50%"]),
            ("How often do you use credit cards or take loans to cover regular expenses?", ["Rarely", "Sometimes", "Often"]),
            ("How do you prefer to invest your money?", ["Fixed Deposits & Savings", "Mutual Funds & Stocks", "High-risk investments"]),
            ("If your investments drop 20% in value, how would you react?", ["Sell immediately", "Wait and watch", "Invest more"]),
            ("Do you have an emergency fund covering at least 6 months of expenses?", ["Yes", "No"]),
            ("Have you ever missed a credit card or loan payment in the last 12 months?", ["Yes", "No"])
        ]
        
        self.answer_scores = {
            0: {"Salary": 5, "Business": 7, "Investments": 10, "Others": 3},
            1: {"Very stable": 10, "Somewhat stable": 6, "Unstable": 2},
            2: {"Less than 10%": 2, "10-30%": 6, "More than 30%": 10},
            3: {"Yes, multiple loans": 2, "Yes, but manageable": 6, "No": 10},
            4: {"Less than 20%": 10, "20-50%": 5, "More than 50%": 2},
            5: {"Rarely": 10, "Sometimes": 5, "Often": 2},
            6: {"Fixed Deposits & Savings": 2, "Mutual Funds & Stocks": 6, "High-risk investments": 10},
            7: {"Sell immediately": 2, "Wait and watch": 6, "Invest more": 10},
            8: {"Yes": 10, "No": 2},
            9: {"Yes": 2, "No": 10}
        }

    def generate_portfolio_allocation(self, risk_score: int) -> Dict:
        if risk_score < 40:  # Low Risk
            high_risk = random.randint(5, 15)
            medium_risk = random.randint(20, 35)
            low_risk = 100 - high_risk - medium_risk
        elif risk_score < 70:  # Moderate Risk
            medium_risk = random.randint(40, 55)
            high_risk = random.randint(20, 35)
            low_risk = 100 - high_risk - medium_risk
        else:  # High Risk
            high_risk = random.randint(50, 65)
            medium_risk = random.randint(25, 35)
            low_risk = 100 - high_risk - medium_risk

        return {
            "high_risk_investments": {
                "percentage": high_risk,
                "breakdown": {
                    "stock_market": round(high_risk * 0.33),
                    "cryptocurrencies": round(high_risk * 0.25),
                    "derivatives": round(high_risk * 0.17),
                    "startup_investments": round(high_risk * 0.17),
                    "commodities": round(high_risk * 0.08)
                }
            },
            "medium_risk_investments": {
                "percentage": medium_risk,
                "breakdown": {
                    "mutual_funds": round(medium_risk * 0.33),
                    "etfs": round(medium_risk * 0.27),
                    "dividend_stocks": round(medium_risk * 0.20),
                    "real_estate": round(medium_risk * 0.13),
                    "corporate_bonds": round(medium_risk * 0.07)
                }
            },
            "low_risk_investments": {
                "percentage": low_risk,
                "breakdown": {
                    "fixed_income": round(low_risk * 0.40),
                    "government_bonds": round(low_risk * 0.30),
                    "pension_funds": round(low_risk * 0.20),
                    "insurance_investments": round(low_risk * 0.10)
                }
            }
        }

    def get_risk_category(self, score: int) -> str:
        if score < 40:
            return "Low Risk Tolerance"
        elif score < 70:
            return "Moderate Risk Tolerance"
        else:
            return "High Risk Tolerance"

    def generate_conversation(self) -> Tuple[List[Dict], Dict]:
        conversation = []
        scores = {}
        
        for idx, (question, options) in enumerate(self.questions):
            # Assistant asks question
            conversation.append({
                "role": "assistant",
                "content": question
            })
            
            # Human responds
            answer = random.choice(options)
            conversation.append({
                "role": "human",
                "content": answer
            })
            
            # Calculate score
            scores[f"q{idx+1}"] = self.answer_scores[idx][answer]

        total_score = sum(scores.values())
        risk_category = self.get_risk_category(total_score)
        
        return conversation, {
            "scores": scores,
            "total_score": total_score,
            "risk_category": risk_category
        }

    def generate_dataset(self, num_profiles: int = 10000) -> List[Dict]:
        datasets = []
        
        for _ in range(num_profiles):
            conversation, risk_calc = self.generate_conversation()
            
            dataset = {
                "risk_category": risk_calc["risk_category"],
                "conversation": conversation,
                "risk_calculation": risk_calc,
                "portfolio_allocation": self.generate_portfolio_allocation(risk_calc["total_score"])
            }
            
            datasets.append(dataset)
            
        return datasets

# Generate the datasets
generator = RiskProfileGenerator()
datasets = generator.generate_dataset(10000)

# Save to file
with open('risk_profiles_dataset.json', 'w') as f:
    json.dump(datasets, f, indent=2)