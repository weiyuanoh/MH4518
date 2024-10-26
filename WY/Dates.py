import pandas as pd 
import pandas_market_calendars as cal 

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
 

def get_early_observation_dates():
    """
    Given start date and end data

    Returns:
        list of early observation dates between start date and end date
    """
    pass