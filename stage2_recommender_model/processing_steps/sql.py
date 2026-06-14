step0_query = """
select *
from raw_orders
where "orderDate"::timestamp >= timestamp '{start_date}' and "orderDate"::timestamp < timestamp '{end_date}'
"""
