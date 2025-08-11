import random
import json
from typing import Dict, List, Union
from copy import deepcopy

class InvestmentProfileGenerator:
    def __init__(self):
        # Previous initialization code remains the same until generate_portfolio_allocation
        self.questions = [
            ("What is your primary source of income?", ["Salary", "Business", "Investments", "Others"]),
            ("How stable is your income?", ["Very stable", "Somewhat stable", "Unstable"]),
            ("What percentage of your income do you save or invest each month?", ["Less than 10%", "10-30%", "More than 30%"]),
            ("Do you have any outstanding loans or EMIs?", ["Yes, multiple loans", "Yes, but manageable", "No"]),
            ("What percentage of your monthly income goes towards loan payments?", ["Less than 20%", "20-50%", "More than 50%"]),
            ("How often do you use credit cards for regular expenses?", ["Rarely", "Sometimes", "Often"]),
            ("How do you prefer to invest your money?", ["Fixed Deposits & Savings", "Mutual Funds & Stocks", "High-risk investments"]),
            ("If your investments drop 20% in value, how would you react?", ["Sell immediately", "Wait and watch", "Invest more"]),
            ("Do you have an emergency fund covering 6 months of expenses?", ["Yes", "No"]),
            ("Have you ever missed a payment in the last 12 months?", ["Yes", "No"])
        ]
        
        self.score_mappings = {
            0: {"Salary": 7, "Business": 8, "Investments": 10, "Others": 5},
            1: {"Very stable": 10, "Somewhat stable": 7, "Unstable": 4},
            2: {"Less than 10%": 4, "10-30%": 7, "More than 30%": 10},
            3: {"Yes, multiple loans": 4, "Yes, but manageable": 7, "No": 10},
            4: {"Less than 20%": 10, "20-50%": 6, "More than 50%": 3},
            5: {"Rarely": 10, "Sometimes": 6, "Often": 3},
            6: {"Fixed Deposits & Savings": 4, "Mutual Funds & Stocks": 7, "High-risk investments": 10},
            7: {"Sell immediately": 3, "Wait and watch": 7, "Invest more": 10},
            8: {"Yes": 10, "No": 4},
            9: {"No": 10, "Yes": 3}
        }

        self.investment_categories = {
            "high_risk_investments": {
                "stock_market": {"min": 20, "max": 30},
                "derivatives": {"min": 10, "max": 15},
                "startup_investments": {"min": 5, "max": 10},
                "commodities": {"min": 5, "max": 10},
                "cryptocurrencies": {"min": 5, "max": 10},
                "forex_trading": {"min": 5, "max": 10},
                "hedge_funds": {"min": 5, "max": 10}
            },
            "medium_risk_investments": {
                "mutual_funds": {"min": 10, "max": 20},
                "etfs": {"min": 5, "max": 15},
                "dividend_stocks": {"min": 5, "max": 10},
                "real_estate": {"min": 3, "max": 8},
                "corporate_bonds": {"min": 2, "max": 7},
                "peer_to_peer_lending": {"min": 2, "max": 7},
                "balanced_funds": {"min": 3, "max": 8}
            },
            "low_risk_investments": {
                "fixed_income": {"min": 5, "max": 15},
                "government_bonds": {"min": 5, "max": 15},
                "pension_funds": {"min": 3, "max": 8},
                "insurance_investments": {"min": 2, "max": 7},
                "savings_accounts": {"min": 2, "max": 7},
                "certificates_of_deposit": {"min": 2, "max": 7},
                "treasury_bills": {"min": 2, "max": 7}
            }
        }

        self.adjustment_requests = [
            "I don't want to invest in cryptocurrencies",
            "I want to focus more on real estate",
            "I prefer not to invest in derivatives",
            "Can we increase allocation to dividend stocks?",
            "I want to avoid startup investments",
            "Please add more government bonds",
            "I want to reduce exposure to commodities",
            "Can we include peer-to-peer lending?",
            "I want to focus on ETFs instead of mutual funds",
            "Please remove forex trading from my portfolio"
        ]

    def get_risk_category(self, score: int) -> str:
        if score < 60:
            return "Low Risk Tolerance"
        elif score < 75:
            return "Moderate Risk Tolerance"
        else:
            return "High Risk Tolerance"

    def generate_portfolio_allocation(self, risk_category: str, score: int) -> Dict:
        base_allocations = {
            "Low Risk Tolerance": {
                "high_risk": 10,
                "medium_risk": 30,
                "low_risk": 60
            },
            "Moderate Risk Tolerance": {
                "high_risk": 30,
                "medium_risk": 45,
                "low_risk": 25
            },
            "High Risk Tolerance": {
                "high_risk": 50,
                "medium_risk": 35,
                "low_risk": 15
            }
        }

        base = base_allocations[risk_category]
        adjustment = (score % 25) / 100

        # Calculate adjusted percentages
        if risk_category == "Low Risk Tolerance":
            low_risk = round(base["low_risk"] + adjustment * 10)
            medium_risk = round(base["medium_risk"] - adjustment * 5)
            high_risk = round(base["high_risk"] - adjustment * 5)
        elif risk_category == "Moderate Risk Tolerance":
            medium_risk = round(base["medium_risk"] + adjustment * 10)
            high_risk = round(base["high_risk"] - adjustment * 5)
            low_risk = round(base["low_risk"] - adjustment * 5)
        else:
            high_risk = round(base["high_risk"] + adjustment * 10)
            medium_risk = round(base["medium_risk"] - adjustment * 5)
            low_risk = round(base["low_risk"] - adjustment * 5)

        portfolio = {}
        for category, total_percent in [("high_risk_investments", high_risk),
                                      ("medium_risk_investments", medium_risk),
                                      ("low_risk_investments", low_risk)]:
            products = self.investment_categories[category]
            selected_products = random.sample(list(products.keys()), 
                                           k=random.randint(3, len(products)))
            
            breakdown = {}
            remaining_percent = total_percent
            
            for product in selected_products[:-1]:
                min_val = products[product]["min"]
                max_val = min(products[product]["max"], remaining_percent - (len(selected_products) - 1))
                
                # Convert to integer and ensure max_val is greater than min_val
                min_val = round(min_val)
                max_val = round(max_val)
                
                if max_val < min_val:
                    percent = min_val
                else:
                    percent = random.randint(min_val, max_val)
                
                breakdown[product] = percent
                remaining_percent -= percent
            
            # Assign remaining percentage to last product
            breakdown[selected_products[-1]] = max(1, round(remaining_percent))
            
            portfolio[category] = {
                "percentage": round(total_percent),
                "breakdown": breakdown
            }

        return portfolio

    def adjust_portfolio(self, original_portfolio: Dict, request: str) -> Dict:
        portfolio = deepcopy(original_portfolio)
        
        def redistribute_percentage(category: Dict, excluded_product: str) -> Dict:
            if excluded_product in category["breakdown"]:
                percent_to_redistribute = category["breakdown"][excluded_product]
                del category["breakdown"][excluded_product]
                
                if category["breakdown"]:
                    per_product = round(percent_to_redistribute / len(category["breakdown"]))
                    for product in category["breakdown"]:
                        category["breakdown"][product] = round(category["breakdown"][product] + per_product)
            return category

        def increase_product(category: Dict, product: str, increase_by: int) -> Dict:
            if product in category["breakdown"]:
                current = category["breakdown"][product]
                to_reduce = round(increase_by / (len(category["breakdown"]) - 1))
                category["breakdown"][product] = round(current + increase_by)
                
                for other_product in category["breakdown"]:
                    if other_product != product:
                        category["breakdown"][other_product] = max(1, round(
                            category["breakdown"][other_product] - to_reduce
                        ))
            return category

        if "don't want" in request.lower() or "remove" in request.lower():
            product = next((p for p in request.lower().split() if p in str(portfolio).lower()), None)
            if product:
                for category_name, category_data in portfolio.items():
                    portfolio[category_name] = redistribute_percentage(category_data, product)
        
        elif "focus more" in request.lower() or "increase" in request.lower():
            product = next((p for p in request.lower().split() if p in str(portfolio).lower()), None)
            if product:
                for category_name, category_data in portfolio.items():
                    portfolio[category_name] = increase_product(category_data, product, 5)

        return portfolio

    def generate_conversation(self) -> Dict:
        conversation = []
        answers = []
        scores = {}
        
        for idx, (question, options) in enumerate(self.questions):
            conversation.append({
                "role": "assistant",
                "content": question
            })
            
            answer = random.choice(options)
            answers.append(answer)
            conversation.append({
                "role": "human",
                "content": answer
            })
            
            scores[f"q{idx+1}"] = self.score_mappings[idx][answer]
        
        total_score = sum(scores.values())
        risk_category = self.get_risk_category(total_score)
        original_portfolio = self.generate_portfolio_allocation(risk_category, total_score)
        
        num_adjustments = random.randint(1, 3)
        portfolio_adjustments = []
        current_portfolio = original_portfolio
        
        for _ in range(num_adjustments):
            request = random.choice(self.adjustment_requests)
            adjusted_portfolio = self.adjust_portfolio(current_portfolio, request)
            portfolio_adjustments.append({
                "user_request": request,
                "adjusted_portfolio": adjusted_portfolio
            })
            current_portfolio = adjusted_portfolio

        return {
            "conversation": conversation,
            "risk_calculation": {
                "scores": scores,
                "total_score": total_score,
                "risk_category": risk_category
            },
            "portfolio_allocation": original_portfolio,
            "portfolio_adjustments": portfolio_adjustments
        }

    def generate_dataset(self, num_profiles: int) -> List[Dict]:
        return [self.generate_conversation() for _ in range(num_profiles)]

# Generate the dataset
if __name__ == "__main__":
    generator = InvestmentProfileGenerator()
    dataset = generator.generate_dataset(10000)

    # Save to file
    with open('investment_profiles.json', 'w') as f:
        json.dump(dataset, f, indent=2)