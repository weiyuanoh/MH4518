import pandas as pd 
import pandas_market_calendars as cal 
import constants as cs 

def num_business_days(startdate, enddate):
    """ 
    given the start date and end date, the number of business days in the SWISS exchange is calculated 
    
    all parameters are timezone naive 
    Parameters:
        startdate (pd.timestamp) : the start date to be used in calculation
        enddate (pd.timestamp): the end date to be used in calculation 
        
    Returns:
        (int) : number of business days in timezone neutral using timezone neutral dates    
    """
    #converting dates to timezone naive
    startdate = pd.Timestamp(startdate).tz_localize(None)
    enddate = pd.Timestamp(enddate).tz_localize(None)
    swiss_cal = cal.get_calendar("SIX")
    n_busi_days = len(swiss_cal.valid_days(startdate, enddate))
    return n_busi_days
 

def get_early_observation_dates(initialdate = cs.initial_fixing_date, enddate = cs.final_fixing_date):
    """
    Given start date and end data

    Returns:
        list of early observation dates between start date and end date
    """
    first_observation_date = initialdate + pd.DateOffset(months=6)
    redemption_dates = pd.date_range(start=first_observation_date, end=enddate, freq='Q')
    redemption_dates = redemption_dates.to_list()

    return redemption_dates


def add_business_days(date: pd.Timestamp):
    # Get the valid trading days starting from the next day
    six_cal = cal.get_calendar('SIX')
    valid_days = six_cal.valid_days(start_date=date + pd.tseries.offsets.BusinessDay(1), end_date=date + pd.Timedelta(days=10))
    
    # Return the first valid day
    return valid_days[0]