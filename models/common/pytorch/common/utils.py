#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#! /usr/bin/env python3
# coding: utf-8

"""
Utils functions for parsers
"""

import sys
from email.message import EmailMessage
from smtplib import SMTP, SMTPException


def is_digit(value):
    """Check if value is digit."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_sorted_filelist(filelist, expected_result_dict):
    """Get sorted filelist."""
    sorted_filelist = []
    if filelist is None:
        for key in expected_result_dict:
            file_name = key + ".log"
            sorted_filelist.append(file_name)
    else:
        sorted_filelist = filelist
    return sorted_filelist


def send_report(sender, recipients, subject, html):
    """Send report."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipients
    msg.set_content(html, subtype="html")
    try:
        with SMTP("smtp.intel.com") as session:
            session.sendmail(sender, recipients, msg.as_string())
    except SMTPException as e:
        raise e


def check_python_version():
    """Check python version."""
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 6):
        print("Python 3.6 or higher is required.")
        print(
            "You are using Python {}.{}.".format(
                sys.version_info.major, sys.version_info.minor
            )
        )
        sys.exit(1)
