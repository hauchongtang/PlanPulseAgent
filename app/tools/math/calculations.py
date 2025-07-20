"""
Math tools for calculation and mathematical operations.
"""

import math
import operator
from typing import Union
from langchain_core.tools import tool


@tool("calculate")
def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    Supports basic arithmetic, math functions like sin, cos, sqrt, etc.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "sqrt(16)", "sin(0)")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Define safe functions for evaluation
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            # Math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "ceil": math.ceil,
            "floor": math.floor,
            "pi": math.pi,
            "e": math.e,
            # Basic operators
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "//": operator.floordiv,
            "%": operator.mod,
            "**": operator.pow,
        }
        
        # Evaluate the expression safely
        result = eval(expression, safe_dict)
        return str(result)
        
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool("convert_units")
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    
    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., 'kg', 'm', 'celsius', 'miles')
        to_unit: Target unit (e.g., 'lbs', 'ft', 'fahrenheit', 'km')
        
    Returns:
        The converted value with units
    """
    try:
        conversions = {
            # Weight
            ("kg", "lbs"): lambda x: x * 2.20462,
            ("lbs", "kg"): lambda x: x / 2.20462,
            ("g", "oz"): lambda x: x * 0.035274,
            ("oz", "g"): lambda x: x / 0.035274,
            
            # Length
            ("m", "ft"): lambda x: x * 3.28084,
            ("ft", "m"): lambda x: x / 3.28084,
            ("km", "miles"): lambda x: x * 0.621371,
            ("miles", "km"): lambda x: x / 0.621371,
            ("cm", "inches"): lambda x: x * 0.393701,
            ("inches", "cm"): lambda x: x / 0.393701,
            
            # Temperature
            ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
            ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
            ("celsius", "kelvin"): lambda x: x + 273.15,
            ("kelvin", "celsius"): lambda x: x - 273.15,
        }
        
        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        else:
            return f"Conversion from {from_unit} to {to_unit} not supported"
            
    except Exception as e:
        return f"Error converting units: {str(e)}"


@tool("percentage_calculator")
def percentage_calculator(operation: str, value1: float, value2: float = None) -> str:
    """
    Calculate percentages and percentage-related operations.
    
    Args:
        operation: Type of calculation ('percent_of', 'increase', 'decrease', 'is_what_percent')
        value1: First value
        value2: Second value (if needed)
        
    Returns:
        The calculated result
    """
    try:
        if operation == "percent_of" and value2 is not None:
            # What is value1% of value2?
            result = (value1 / 100) * value2
            return f"{value1}% of {value2} = {result}"
            
        elif operation == "increase" and value2 is not None:
            # Increase value1 by value2%
            result = value1 * (1 + value2 / 100)
            return f"{value1} increased by {value2}% = {result}"
            
        elif operation == "decrease" and value2 is not None:
            # Decrease value1 by value2%
            result = value1 * (1 - value2 / 100)
            return f"{value1} decreased by {value2}% = {result}"
            
        elif operation == "is_what_percent" and value2 is not None:
            # value1 is what percent of value2?
            result = (value1 / value2) * 100
            return f"{value1} is {result}% of {value2}"
            
        else:
            return f"Invalid operation '{operation}' or missing required values"
            
    except Exception as e:
        return f"Error calculating percentage: {str(e)}"
