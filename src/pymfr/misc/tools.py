'''

@author: "epsilonyuan@gmail.com"
'''

class FieldProxy:
    def __init__(self, field_name, proxy_name):
        self.field_name = field_name
        self.proxy_name = proxy_name
        
    def __get__(self, obj, type=None):
        field = getattr(obj, self.field_name)
        return getattr(field, self.proxy_name)
    
    def __set__(self, obj, value):
        field = getattr(obj, self.field_name)
        setattr(field, self.proxy_name, value)