# -*- coding: utf-8 -*-
"""
Created on Sun May 14 02:12:51 2023

@author: mr.laptop
"""

from pydantic import BaseModel
class Similarity(BaseModel):
    inpImg: list[str]
    proImg: list[str]
    

    
