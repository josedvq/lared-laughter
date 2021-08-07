def get_consent_task():
    return {
        "name": "Consent",
        "id": "consent",
        "type": "InstructionsTask",
        "prerequisite": True,
        "content": {
            "type": "link",
            "url": "$$www$$/consent.md"
        },
        "form": {
            "fields": [
                {
                    "name": "consent1",
                    "label": "I have read and understood the study information dated [DD/MM/YYYY], or it has been read to me. I have been able to ask questions about the study and my questions have been answered to my satisfaction.",
                    "required": True,
                    "input": {
                        "inputType": "Checkbox.Group",
                        "options": [
                            {
                                "label": "Yes",
                                "value": "yes"
                            }
                        ]
                    }
                },
                {
                    "name": "consent2",
                    "label": "I consent voluntarily to be a participant in this study and understand that I can refuse to answer questions and I can withdraw from the study at any time, without having to give a reason. ",
                    "required": True,
                    "input": {
                        "inputType": "Checkbox.Group",
                        "options": [
                            {
                                "label": "Yes",
                                "value": "yes"
                            }
                        ]
                    }
                },
                {
                    "name": "consent3",
                    "label": "I understand that taking part in the study involves providing data ratings or annotations which will be recorded but not linked to my identity, along with data about my interaction with the web interface.",
                    "required": True,
                    "input": {
                        "inputType": "Checkbox.Group",
                        "options": [
                            {
                                "label": "Yes",
                                "value": "yes"
                            }
                        ]
                    }
                },
                {
                    "name": "consent4",
                    "label": "I understand that information I provide will be used for scientific publications.",
                    "required": True,
                    "input": {
                        "inputType": "Checkbox.Group",
                        "options": [
                            {
                                "label": "Yes",
                                "value": "yes"
                            }
                        ]
                    }
                },
                {
                    "name": "consent5",
                    "label": "I understand that personal information collected about me that can identify me, such as [e.g. my name or where I live], will not be asked nor stored.",
                    "required": True,
                    "input": {
                        "inputType": "Checkbox.Group",
                        "options": [
                            {
                                "label": "Yes",
                                "value": "yes"
                            }
                        ]
                    }
                },
                {
                    "name": "consent6",
                    "label": "I give permission for the annotations that I provide to be archived in networked storage at Delft University of Technology in anonymized form so it can be used for future research. I understand that data will not be made public but may be shared with researchers under an End User License Agreement (EULA).",
                    "required": True,
                    "input": {
                        "inputType": "Checkbox.Group",
                        "options": [
                            {
                                "label": "Yes",
                                "value": "yes"
                            }
                        ]
                    }
                }
            ]
        }
}

def get_general_instructions_task(fname = 'laughter'):
    return {
        "name": "General Instructions",
        "id": "gen_instructions",
        "type": "InstructionsTask",
        "prerequisite": False,
        "content": {
            "type": "link",
            "url": f"$$www$$/{fname}_instructions.md"
        },
        "maxSubmissions": 1
    }

def get_condition_instructions_task(file, type='video'):
    titles = {
        'video': 'Video-only instructions',
        'audio': 'Audio-only instructions',
        'av': 'Audiovisual instructions'
    }
    return {
        "name": titles[type],
        "id": f"{type}_instructions",
        "type": "InstructionsTask",
        "prerequisite": False,
        "content": {
            "type": "link",
            "url": f"$$www$$/{file}"
        },
        "maxSubmissions": 1
    }

def get_experience_rating():
    return {
        "name": "Feedback",
        "type": "InstructionsTask",
        "content": {
            "type": "link",
            "url": f"$$www$$/feedback.md"
        },
        "form": {
            "fields": [
                {
                    "name": "rating",
                    "label": "How would you rate your experience in completing this experiment?",
                    "input": {
                        "inputType": "Rate",
                    }
                },{
                    "name": "feedback",
                    "label": "Do you have any comments about the process? Did you find it frustrating, tiring or too long? Were the instructions clear? Did you have any issues with the tool?",
                    "input": {
                        "inputType": "Input.TextArea",
                    }
                }
            ]
        },
    }