# %%
import pandas as pd

accetped_df = pd.read_csv("Loan_status_2007-2020Q3.csv")
# rejected_df = pd.read_csv("rejected_2007_to_2018Q4.csv")

# %%
columns = [
    "id",
    "member_id",
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",  # The self-reported annual income provided by the borrower during registration.
    "verification_status",  # Indicates if income was verified by LC, not verified, or if the income source was verified
    "issue_d",
    "loan_status",
    "pymnt_plan",
    "url",  # URL for the LC page with listing data.
    "desc",
    "purpose",
    "title",
    "zip_code",
    "addr_state",  # The state provided by the borrower in the loan application
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",  # post charge off collection fee
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "collections_12_mths_ex_med",  # Number of collections in 12 months excluding medical collections
    "mths_since_last_major_derog",
    "policy_code",
    "application_type",  # Indicates whether the loan is an individual application or a joint application with two co-borrowers
    "annual_inc_joint",  # The combined self-reported annual income provided by the co-borrowers during registration
    "dti_joint",
    "verification_status_joint",
    "acc_now_delinq",  # The number of accounts on which the borrower is now delinquent.
    "tot_coll_amt",  #
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",  # Balance to credit limit on all trades
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",  # Number of trades opened in past 24 months.
    "avg_cur_bal",  # Average current balance of all accounts
    "bc_open_to_buy",  # Total open to buy on revolving bankcards.
    "bc_util",  # Ratio of total current balance to high credit/credit limit for all bankcard accounts.
    "chargeoff_within_12_mths",  # Number of charge-offs within 12 months
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "disbursement_method",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]

# %%

features = [
    "funded_amnt",
    "emp_length",
    "annual_inc",
    "home_ownership",
    "grade",
    "last_pymnt_amnt",
    "mort_acc",
    "pub_rec",
    "int_rate",
    "open_acc",
    "num_actv_rev_tl",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_old_rev_tl_op",
    "bc_util",
    "bc_open_to_buy",
    "avg_cur_bal",
    "acc_open_past_24mths",
    "loan_status", # https://help.lendingclub.com/hc/en-us/articles/216109367-What-Do-the-Different-Note-Statuses-Mean
]

neg = accetped_df["loan_status"] in ["Charged Off", "Does not meet the credit policy. Status:Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Default"]
pos = accetped_df["loan_status"] in ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]


def heuristical_match():
    pass
