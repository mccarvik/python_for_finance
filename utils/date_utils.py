import datetime

def dcc_calc(dcc, start, mid, end):
    if dcc == "ACT/ACT":
        delta_num = mid - start
        delta_den = end - start
        return tuple(["{0}/{1}".format(delta_num.days, delta_den.days), (delta_num.days / delta_den.days)])
    elif dcc == "ACT/360":
        delta_num = mid - start
        days_front = 30 - start.day if 30 - start.day > 0 else 0
        days_end = end.day
        days_mid = (end.month - start.month - 1) * 30
        days_mid = 0 if days_mid < 0 else days_mid
        delta_den = days_end + days_mid + days_front
        return tuple(["{0}/{1}".format(delta_num.days, delta_den), (delta_num.days / delta_den)])
    elif dcc == "30/360":
        days_front_num = 30 - start.day if 30 - start.day > 0 else 0
        days_end_num = mid.day
        days_mid_num = (mid.month - start.month - 1) * 30
        days_mid_num = 0 if days_mid_num < 0 else days_mid_num
        delta_num = days_end_num + days_mid_num + days_front_num
        
        days_front_den = 30 - start.day if 30 - start.day > 0 else 0
        days_end_den = end.day
        days_mid_den = (end.month - start.month - 1) * 30
        days_mid_den = 0 if days_mid_den < 0 else days_mid_den
        delta_den = days_end_den + days_mid_den + days_front_den
        return tuple(["{0}/{1}".format(delta_num, delta_den), (delta_num / delta_den)])

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    s = datetime.date(2012, 3, 1)
    m = datetime.date(2012, 7, 3)
    e = datetime.date(2012, 9, 1)
    print(dcc_calc("ACT/ACT", s, m, e))
    print(dcc_calc("ACT/360", s, m, e))
    print(dcc_calc("30/360", s, m, e))