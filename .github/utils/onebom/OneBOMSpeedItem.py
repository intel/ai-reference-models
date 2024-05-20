import json
from OneBOMGlobals import *


class OneBOMSpeedItem:
  AIRM_FAMILY_ITEM_CODE = 10054461
  AIRM_WORKLOAD_COLLECTION_VARIANT_ITEM_CODE = 10058363
  RELEASE = "Release"
  VARIANT = "Variant"
  BOM_ITEMS = "BOMItems"
  _speed_item_json = None

  def __init__(self, speed_item_type, parent_item_code, app_path, release_info="" ):
    """
    Initializes a new instance of the OneBOMSpeedItem class.

    Args:
      speed_item_type (str): The speed item type.
      parent_item_code (str): The parent item code.
      app_path (str): The app path.
      release_info (str, optional): The release information. Defaults to "".
    """
    self.app_path = app_path
    self.speed_item_type = speed_item_type
    self.parent_item_code = parent_item_code
    self.release_info = ""
    if release_info is not None and release_info != "":
      self.release_info = release_info + " "
    self._speed_item_json = {
      "AppName": "PLM",
      "BatchID": 16285726,
      "Wwid": "10637375",
      "Items": [
        {
          "RowId": "1",
          "CrudType": "Create",
          "IngredientSKU": f"{AIRM_PREFIX} {self.app_path} {self.release_info}{self.speed_item_type}",
          "ItemDsc": f"{AIRM_PREFIX} {self.app_path} {self.release_info}{self.speed_item_type}",
          "MaterialTypeCd": "PLIN",
          "ClassDsc": "DSGN_PROD_LINE_SW",
          "CreateWwid": 10637375,
          "OrganizationUnitCd":"02039",
          "NoteTxt": "",
          "BusinessUnitNm": "DESIGN_ITEM",
          "ParentItemCd": f"{self.parent_item_code}",
          "ItemAttributeValues": [
            {
              "AttributeNm": "INGREDIENT_TYPE",
              "StatusNm": "SPEED UDA",
              "AttributeVal":f"{self.speed_item_type}"
            },
            {
              "CrudType": "CREATE",
              "AttributeNm": "DESIGN_ITEM_TYPE",
              "StatusNm": "SPEED UDA",
              "AttributeVal": "Software"
            },
            {
              "CrudType": "CREATE",
              "AttributeNm": "DESIGN_ITEM_DESCRIPTION",
              "StatusNm": "SPEED UDA",
              "AttributeVal": f"Scripts to run the AI workload found at {self.app_path}."
            },
            {
              "AttributeNm": "DESIGN_ITEM_STATUS",
              "StatusNm": "SPEED UDA",
              "AttributeVal": "Active"
            },
            {
              "CrudType": "CREATE",
              "AttributeNm": "SW_CLASS",
              "StatusNm": "SPEED UDA",
              "AttributeVal": "Application"
            },
            {
              "CrudType": "CREATE",
              "AttributeNm": "SOURCING",
              "StatusNm": "SPEED UDA",
              "AttributeVal": "INTEL_MAKE"
            }
          ]
        }
      ]
    }

  def dumps(self):
    """
    Returns the JSON representation of the speed item.

    Returns:
      str: The JSON representation of the speed item.
    """
    return json.dumps(self._speed_item_json)

class OneBOMSpeedVariant(OneBOMSpeedItem):
  def __init__(self, app_path ):
    super().__init__(OneBOMSpeedItem.VARIANT, OneBOMSpeedItem.AIRM_FAMILY_ITEM_CODE, app_path)

class OneBOMSpeedRelease(OneBOMSpeedItem):
  def __init__(self, parent_item_code, app_path, release_info):
    super().__init__(OneBOMSpeedItem.RELEASE, parent_item_code, app_path, release_info)

class OneBOMSpeedReleaseBOMItem(OneBOMSpeedItem):
  def __init__(self, parent_item_code, bom_item_code, release_info=""):
    super().__init__(None, None, None, None)
    self._speed_item_json = {
      "AppName": "PLM",
      "BatchId": "123",
      "Wwid": "10637375",
      f"{OneBOMSpeedItem.BOM_ITEMS}": [
        {
          "RowId": "1",
          "CrudType": "CREATE",
          "ParentItemCd": f"{parent_item_code}",
          "ChildItemCd": f"{bom_item_code}",
          "ChildQty": "1"    
        }
      ]
    } 
    
