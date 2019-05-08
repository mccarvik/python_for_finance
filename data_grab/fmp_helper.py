import pdb

BAL_SHEET_MAP = {
    'Accounts payable': 'accounts_payable',
    'Accrued liabilities': 'accrued_liabs',
    'Accrued expenses and liabilities': 'accrued_liabs',
    'accrued expenses and liabilities': 'accrued_liabs',
    'Accumulated Depreciation': 'accumulated_depr',
    'Accumulated depreciation': 'accumulated_depr',
    'Accumulated other comprehensive income': 'accumulated_other_comp_inc',
    'Cash and cash equivalents': 'cash',
    'Common stock': 'stock',
    'Deferred income taxes': 'def_tax_liabs',
    'Deferred taxes': 'def_tax_liabs',
    'Deferred revenues': 'def_revenue',
    'Deferred taxes liabilities': 'def_tax_liabs',
    'Deferred income tax assets': 'def_inc_tax_assets',
    'Deferred policy acquisition costs': 'def_policy_acq_costs',
    'Equity and other investments': 'eq_and_other_inv',
    'Goodwill': 'goodwill',
    'Gross property, plant and equipment': 'gross_ppe',
    'Property and equipment, at cost': 'gross_ppe',
    'Intangible assets': 'int_assets',
    'Inventories': 'inv',
    'Long-term debt': 'long_term_debt',
    'Net property, plant and equipment': 'net_ppe',
    'Property, plant and equipment, net': 'net_ppe',
    'Property and equipment': 'net_ppe',
    'Other current assets': 'other_cur_assets',
    'Other current liabilities': 'other_cur_liabs',
    'Other long-term assets': 'other_long_term_assets',
    'Other long-term liabilities': 'other_long_term_liabs',
    'Receivables': 'receivables',
    'Retained earnings': 'retained_earnings',
    'Short-term debt': 'short_term_debt',
    'Short-term investments': 'short_term_inv',
    'Taxes payable': 'tax_payable',
    'Income taxes payable': 'tax_payable',
    'Total assets': 'total_assets',
    'Total cash': 'total_cash',
    'Total current assets': 'total_cur_assets',
    'Total current liabilities': 'total_cur_liabs',
    'Total liabilities': 'total_liabs',
    "Total liabilities and stockholders' equity": 'total_liabs_and_eq',
    'Total non-current assets': 'total_non_cur_assets',
    'Total non-current liabilities': 'total_noncur_liabs',
    "Total stockholders' equity": 'total_equity',
    "Total Stockholders' equity": 'total_equity',
    'Additional paid-in capital': 'add_paid_in_cap',
    'Minority interest': 'minority_int',
    'Minority Interest': 'minority_int',
    'Prepaid expenses': 'prepaid_exp',
    'Treasury stock': 'tsy_stock',
    'Capital leases': 'cap_leases',
    'Preferred stock': 'pref_stock',
    'Allowance for loan losses': 'allow_for_loan_loss',
    'Cash and due from banks': 'cash_due_from_banks',
    'Debt securities': 'debt_securities',
    'Fixed maturity securities': 'fixed_mat_securities',
    'Equity securities': 'eq_securities',
    'Deposits': 'deposits',
    'Deposits with banks': 'deposits_with_banks',
    'Derivative assets': 'deriv_assets',
    'Derivative liabilities': 'deriv_liabs',
    'Federal funds purchased': 'fed_funds_purchased',
    'Federal funds sold': 'fed_funds_sold',
    'Loans': 'loans',
    'Loans, total': 'loans_total',
    'Net loans': 'net_loans',
    'Payables': 'payables',
    'Payables and accrued expenses': 'payables',
    'Premises and equipment': 'fixtures_and_equip',
    'Fixtures and equipment': 'fixtures_and_equip',
    'Short-term borrowing': 'short_term_borrow',
    'Trading assets': 'trading_assets',
    'Trading liabilities': 'trading_liabs',
    'Other Equity': 'other_equity',
    'Other equity': 'other_equity',
    'Other assets': 'other_assets',
    'Other intangible assets': 'other_int_assets',
    'Other liabilities': 'other_liabs',
    'Other liabilities and equities': 'other_liabs_and_eq',
    'Other properties': 'other_props',
    'Real estate properties, net': 'other_props',
    'Real estate': 'other_props',
    'Real estate properties': 'other_props',
    'Accrued investment income': 'accrued_inv_inc',
    'Future policy benefits': 'future_policy_bens',
    'Pensions and other benefits': 'pensions_and_other_bens',
    'Prepaid pension benefit': 'pensions_and_other_bens',
    'Prepaid pension costs': 'pensions_and_other_bens',
    'Pensions and other postretirement benefits': 'pensions_and_other_bens',
    'Policyholder funds': 'policyholder_funds',
    'Premiums and other receivables': 'prem_and_other_receivables',
    'Separate account assets': 'sep_account_assets',
    'Separate account liabilities': 'sep_account_liabs',
    'Unearned premiums': 'unearned_prems',
    'Restricted cash and cash equivalents': 'restrict_cash_and_equiv',
    'Restricted cash': 'restrict_cash_and_equiv',
    'Trading securities': 'secs_and_invs',
    'Securities and investments': 'secs_and_invs',
    'Investments': 'secs_and_invs',
    'Securities borrowed': 'secs_borrowed',
    'securities borrowed': 'secs_borrowed',
    'Land': 'land',
    'General Partner': 'gen_partner',
    'Regulatory assets': 'regulatory_assets',
    'Regulatory liabilities': 'regulatory_liabs',
    'Commercial loans': 'comm_loans',
}




COL_MAPS = {
    "bal_sheet": BAL_SHEET_MAP
}

def map_columns(data_set, cols):
    # for cc in cols: 
    #     print("'" + cc + "': '" + cc.lower().replace(" ", "_") + "',")
    col_map = COL_MAPS[data_set]
    new_cols = []
    for col in cols:
        try:
            new_cols.append(col_map[col])
        except Exception as exc:
            print("'" + col + "': '" + col.lower().replace(" ", "_") + "',")
            pdb.set_trace()
            print(exc)
    return new_cols
    