# %%
class OnchainFeatureSet:
    def __init__(self, addr, borrow_amounts, repay_amounts, deposit_amounts, withdraw_amounts) -> None:
        self.borrow_count = len(borrow_amounts)
        self.total_borrow_amount = sum(borrow_amounts)

        self.repay_count = len(repay_amounts)
        self.total_repay_amount = sum(repay_amounts)

        self.deposit_count = len(repay_amounts)
        self.total_deposit_amount = sum(repay_amounts)

        self.withdraw_count = len(repay_amounts)
        self.total_withdraw_amount = sum(repay_amounts)


# %%

