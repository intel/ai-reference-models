#!/usr/bin/env python3

import os
import argparse
from OneBOMRequestManager import *
from OneBOMSpeedItem import *
from OneBOMGlobals import *

class OneBOMSpeedReleaseFactory:
  _release_bom_items = []
  _onebom_reqest_manager = None
  _onebom_speed_request_manager = None
  def run(self):
 
    # Call the method to create variants
    if self.args.apps_needing_variants:
      one_bom_speed_item_factory.create_variants()

    # Call the method to create releases
    if self.args.apps_needing_release:
      one_bom_speed_item_factory.create_releases()

    # Create Workload Collection Release
    if self.args.release_workload_collection:
      one_bom_speed_item_factory.create_workload_collection_release()

  def __init__(self):
    # Create the parser
    parser = argparse.ArgumentParser(description="Creates OneBoM SPEED items from apps-needing-variant.txt and apps-needing-release.txt.")

    # Add the arguments
    parser.add_argument('-v', '--apps-needing-variants', type=str, default=None, required=False, help='path to the apps-needing-variant.txt file')
    parser.add_argument('-r', '--apps-needing-release', type=str, default=None, required=False, help='path to the apps-needing-variant.txt file')
    parser.add_argument('-i', '--release-info', type=str, default=None, help='The release variant information, e.g. v3.1.1')
    parser.add_argument('-d', '--dry-run', action='store_true', help="Don't actually create the items, just print what would be done.")
    parser.add_argument('-t', '--use-token', action='store_true', help='Use OAUTH Token.')
    parser.add_argument('-w', '--release-workload-collection', type=str, default=None, required=False, help='path to the list of completed Release item codes that need to be added to the workload collection')
    parser.add_argument('-l', '--log-level', type=str, default=None, help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    
    self.parse_and_validate_args(parser)
    self.release_parent_item_code = None
    self._onebom_reqest_manager = OneBOMRequestManager()
    self._onebom_speed_reqest_manager = OneBOMSpeedRequestManager()

  def parse_and_validate_args(self, parser):
    # Parse the arguments
    self.args = parser.parse_args()

    if self.args.log_level:
      # If they put anything here, just debug
        DEBUG = True
    if self.args.apps_needing_variants and not os.path.exists(self.args.apps_needing_variants):
      error(f"{self.args.apps_needing_variants} does not exist.")

    if self.args.apps_needing_release and not os.path.exists(self.args.apps_needing_release):
      error(f"{self.args.apps_needing_release} does not exist.")

    if self.args.release_info and self.args.release_info == "":
      error ("release_info is required.")

    if self.args.release_workload_collection:
      if os.path.exists(self.args.release_workload_collection):
        debug(f"Release Workload Collection file: {self.args.release_workload_collection}")
        with open(self.args.release_workload_collection, 'r') as f:
          for line in f:
            self._release_bom_items.append(line.strip())
      else:
        print(f"{self.args.release_workload_collection} does not exist. Generating automatically.")
        
  def create_variant(self, app_path):
    speed_item = OneBOMSpeedVariant(app_path)
    if self.args.dry_run:
      print (f"Would create variant and release for {app_path}")
    else:
      response = self._onebom_speed_reqest_manager.put(OneBOMSpeedRequestManager.ITEM_ENDPOINT, data=speed_item.dumps())
      if response is not None and response.status_code == 200:
        print (f"{app_path} variant: SUCCESS!!!!")
        # get the item code so we can create the release
        item_code = response.json()['Items'][0]['ItemCd']
        debug (f"Item Code: {item_code}")
        # create a release using the item code
        self.create_release(app_path, item_code)

  def get_existing_item_code(self, type, app_path):
    full_path = AIRM_PREFIX + app_path + type
    debug (f"Getting variant item code for {full_path}")
    if type == OneBOMSpeedItem.VARIANT:
      url = f"{OneBOMRequestManager.VARIANT_DETAILS_ENDPOINT}?$format=JSON&ItemFullDsc={full_path}"
    elif type == OneBOMSpeedItem.RELEASE:
      url = f"{OneBOMRequestManager.RELEASE_DETAILS_ENDPOINT}?$format=JSON&ItemFullDsc={full_path}"
    else:
      error (f"Unknown type: {type}")

    response = self._onebom_reqest_manager.get(url)
    if response is not None and response.status_code == 200:
      # get the item code so we can create the release
      item_code = response.json()['elements'][0]['EnterpriseItemId']
      debug (f"SUCCESS!!!! Parent Variant Item Code: {item_code}")
      return item_code
    else:
      error (f"{app_path} FAILURE!!!! {response.text }")
      return None

  def create_variants(self):
    with open(self.args.apps_needing_variants, 'r') as f:
      for line in f:
        app_path = line.strip()[ROOT_STR_LEN:-REQUIREMENTS_STR_LEN]
        if self.args.dry_run:
          print (f"Would create variant for {app_path}")
        else:
          variant_item_code = self.create_variant(app_path)
          # With the new Variant Item code, we need to create a new release for it
          if variant_item_code is not None:
            self.create_release(app_path, variant_item_code)  

  def create_release(self, app_path, parent_item_code):
    speed_item = OneBOMSpeedRelease( parent_item_code, app_path, self.args.release_info)
    response = self._onebom_speed_reqest_manager.put(OneBOMSpeedRequestManager.ITEM_ENDPOINT, data=speed_item.dumps())
    if response is not None and response.status_code == 200:
      print (f"{app_path} release: SUCCESS!!!!")
        # get the item code so we can add it to the collection BoM
      self._release_bom_items.append(response.json()['Items'][0]['ItemCd'])
    else:
      error (f"{app_path} release: FAILURE!!!!")

  def create_releases(self):
    with open(self.args.apps_needing_release, 'r') as f:
      for line in f:
        app_path = line.strip()[ROOT_STR_LEN:-REQUIREMENTS_STR_LEN]

        # call the software-variant-details endpoint to get the EnterpriseItemId
        parent_item_code = self.get_existing_item_code(OneBOMSpeedItem.VARIANT, app_path)
        if self.args.dry_run:
          print (f"Would create release for {app_path} with parent variant code {parent_item_code}")
        else:
          speed_item = OneBOMSpeedRelease(parent_item_code, app_path, self.args.release_info)
          response = self._onebom_speed_reqest_manager.put(OneBOMSpeedRequestManager.ITEM_ENDPOINT, data=speed_item.dumps())
          if response is not None and response.status_code == 200:
            print (f"{app_path} release: SUCCESS!!!!")
          else:
            error (f"{app_path} release: FAILURE!!!!")

  def create_workload_collection_release(self):
    # create a release for the workload collection
    workload_collection_release_item_code = None
    speed_item = OneBOMSpeedRelease(OneBOMSpeedItem.AIRM_WORKLOAD_COLLECTION_VARIANT_ITEM_CODE, 
                                    "", 
                                    self.args.release_info)
    if self.args.dry_run:
      print (f"""Would create workload-collection release {self.args.release_info}:
             {self._release_bom_items}""")
    else:
      response = self._onebom_speed_reqest_manager.put(OneBOMSpeedRequestManager.ITEM_ENDPOINT, data=speed_item.dumps())
      # for now, just hack and gittr done
      workload_collection_release_item_code = 10060455
      # if response is not None and response.status_code == 200:
      #   print (f"workload-collection release: SUCCESS!!!!")
      #   # get the item code so we can create the release
      #   workload_collection_release_item_code = response.json()['Items'][0]['ItemCd']
      # elif response is not None and (response.status_code == 400 and response.text.find("existing Product Line") != -1):
      #   print (f"NOTE: workload-collection release {self.args.release_info} already exists.")
      #   #use another method to get the Workload collection release Item Code
      #   workload_collection_release_item_code= self.get_existing_item_code(OneBOMSpeedItem.RELEASE, 
      #                       f"Intel® AI Reference Models - {self.args.release_info} {OneBOMSpeedItem.RELEASE}")
      #   if workload_collection_release_item_code is None:
      #     error (f"workload-collection release: FAILURE!!!! Status Code: {response.status_code}; Text: {response.text}")
      # else:
      #   error (f"workload-collection release: FAILURE!!!! Status Code: {response.status_code}; Text: {response.text}")

      debug (f"Workload Collection Release {self.args.release_info} Item Code: {workload_collection_release_item_code}")
      # build the BoM using all the releases
      for bom_item_code in self._release_bom_items:
        bom_item = OneBOMSpeedReleaseBOMItem(workload_collection_release_item_code, bom_item_code.split(',')[1])
        response = self._onebom_speed_reqest_manager.post(OneBOMSpeedRequestManager.BOM_ENDPOINT, data=bom_item.dumps())
        if response is not None and response.status_code == 200:
          print (f"workload-collection BoM addition: SUCCESS!!!! {bom_item_code}")
        else:
          error (f"workload-collection BoM: FAILURE!!!! Status: Item Code: {bom_item_code}; {response.status_code}; Text: {response.text}")      

if __name__ == "__main__":
  # Create an instance of the class
  one_bom_speed_item_factory = OneBOMSpeedReleaseFactory()
  one_bom_speed_item_factory.run()
