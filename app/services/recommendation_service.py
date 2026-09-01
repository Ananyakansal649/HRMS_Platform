"""
Enterprise HR AI — Upskilling Recommendation Engine
Maps missing skills to training recommendations (spec task 15).
"""
from app.services.skill_gap_service import compute_employee_skill_gaps


# ---------------------------------------------------------------------------
# Comprehensive skill-to-recommendation mapping for all O*NET skill names
# found in essential_skills.csv and software_skills.csv.
# ---------------------------------------------------------------------------
SKILL_TO_RECOMMENDATION: dict[str, str] = {

    # ── Soft Skills / Cognitive ──────────────────────────────────────────────
    "Active Learning":          "Complete active learning and self-directed professional development training.",
    "Active Listening":         "Take active listening and interpersonal communication skills training.",
    "Critical Thinking":        "Enroll in critical thinking and problem-solving training with practical case-study exercises.",
    "Learning Strategies":      "Complete learning strategies and study-skills training for efficient knowledge acquisition.",
    "Mathematics":              "Take foundational mathematics and quantitative reasoning skills training.",
    "Reading Comprehension":    "Complete reading comprehension and information-analysis skills training.",
    "Speaking":                 "Take professional speaking and presentation skills training.",
    "Writing":                  "Complete business writing and technical documentation skills training.",
    "Science":                  "Complete scientific reasoning and research methodology training.",
    "Monitoring":               "Take workplace monitoring and performance evaluation training.",

    # ── Office / Productivity Software ───────────────────────────────────────
    "Access software":                  "Complete Microsoft Access database management training.",
    "Accounting software":              "Complete accounting software training (QuickBooks, Sage, or similar).",
    "Administration software":          "Take administrative software and office management tools training.",
    "Audit software":                   "Complete audit management software training.",
    "Bar coding software":              "Complete barcode scanning and inventory tracking software training.",
    "Billing and invoicing software":   "Complete billing and invoicing software training for accurate financial processing.",
    "Calendar and scheduling software": "Complete calendar and scheduling tools training (Outlook, Google Calendar).",
    "Charting software":                "Complete charting and data visualization tools training.",
    "Compliance software":              "Complete compliance management software training.",
    "Contact center software":          "Complete contact center and customer service platform training.",
    "Desktop communications software":  "Complete desktop communication tools training (Teams, Slack, Zoom).",
    "Dictionary software":              "Complete digital reference and dictionary tools training.",
    "Document management software":     "Complete document management system (DMS) training.",
    "Electronic mail software":         "Complete professional email and communication tools training.",
    "Fax software":                     "Complete fax and digital document transmission training.",
    "Financial analysis software":      "Complete financial analysis software training (Excel modeling, Bloomberg).",
    "Foreign language software":         "Complete foreign language learning software training.",
    "Graphics or photo imaging software": "Complete graphics and photo editing software training (Photoshop, GIMP).",
    "Instant messaging software":       "Complete instant messaging and team collaboration tools training.",
    "Internet browser software":        "Complete web browser and internet navigation training.",
    "Label making software":            "Complete label design and printing software training.",
    "Legal management software":        "Complete legal management and case-tracking software training.",
    "Library software":                 "Complete library management system training.",
    "License management software":      "Complete license management and compliance tracking training.",
    "Mailing and shipping software":    "Complete mailing and shipping management software training.",
    "Map creation software":            "Complete mapping and geographic visualization tools training.",
    "Medical software":                 "Complete medical records and healthcare software training.",
    "Mobile location based services software": "Complete mobile location-based services and GPS tools training.",
    "Mobile messaging service software":      "Complete mobile messaging platform training.",
    "Mobile operator specific application software": "Complete mobile operator-specific application training.",
    "Multi-media educational software": "Complete multimedia educational tools and e-learning platform training.",
    "Music or sound editing software":  "Complete audio editing and sound production software training (Audacity, GarageBand).",
    "Office suite software":            "Complete Microsoft Office/Google Workspace productivity suite training.",
    "Optical character reader OCR or scanning software": "Complete OCR and document scanning software training.",
    "Presentation software":            "Complete presentation design tools training (PowerPoint, Keynote, Google Slides).",
    "Spell checkers":                   "Complete proofreading and spell-check tools training.",
    "Spreadsheet software":             "Complete spreadsheet software training covering formulas, pivot tables, and data analysis.",
    "Tax preparation software":         "Complete tax preparation software training (TurboTax, H&R Block).",
    "Time accounting software":         "Complete time tracking and timesheet management software training.",
    "Video conferencing software":      "Complete video conferencing tools training (Zoom, Teams, Webex).",
    "Video creation and editing software": "Complete video creation and editing software training (Premiere, Final Cut).",
    "Web page creation and editing software": "Complete web page creation and editing tools training (WordPress, Wix).",
    "Word processing software":         "Complete word processing software training (Word, Google Docs).",

    # ── Data & Analytics Software ────────────────────────────────────────────
    "Business intelligence and data analysis software": "Complete BI and data analytics platform training (Power BI, Tableau, Looker).",
    "Categorization or classification software": "Complete categorization and machine-learning classification tools training.",
    "Analytical or scientific software":  "Complete analytical and scientific software training (MATLAB, SPSS, or similar).",
    "Data base management system software": "Complete database management system training (SQL Server, MySQL, PostgreSQL).",
    "Data base reporting software":       "Complete database reporting and dashboarding tools training.",
    "Data base user interface and query software": "Complete database query and interface tools training (SQL, query builders).",
    "Data compression software":          "Complete data compression and archiving tools training.",
    "Data conversion software":           "Complete data conversion and ETL tools training.",
    "Data mining software":               "Complete data mining and pattern recognition training.",
    "Geographic information system":      "Complete GIS and spatial analysis tools training (ArcGIS, QGIS).",
    "Metadata management software":       "Complete metadata management and data governance tools training.",
    "Object oriented data base management software": "Complete NoSQL and object-oriented database training.",
    "Risk management data and analysis software": "Complete risk management and data analysis tools training.",
    "Inventory management software":      "Complete inventory management and warehouse operations software training.",
    "Procurement software":               "Complete procurement and purchasing management software training.",
    "Time accounting software":           "Complete time tracking and timesheet management software training.",

    # ── Dev / Engineering Tools ──────────────────────────────────────────────
    "Application server software":                "Complete application server and middleware training.",
    "Compiler and decompiler software":           "Complete compiler design and software analysis tools training.",
    "Configuration management software":          "Complete configuration management and DevOps tools training.",
    "Development environment software":           "Complete IDE and development environment tools training.",
    "File versioning software":                   "Complete version control training (Git, SVN).",
    "Object or component oriented development software": "Complete OOP and software engineering design patterns training.",
    "Program testing software":                   "Complete software testing and QA automation tools training.",
    "Requirements analysis and system architecture software": "Complete requirements analysis and system architecture tools training.",
    "Web platform development software":          "Complete web platform and full-stack development tools training.",
    "Process mapping and design software":        "Complete process mapping and business process modeling tools training.",

    # ── Networking / Infrastructure ──────────────────────────────────────────
    "Authentication server software":  "Complete authentication and identity management server training.",
    "Backup or archival software":     "Complete backup, archival, and disaster recovery tools training.",
    "Bridge software":                 "Complete network bridge and interconnectivity software training.",
    "Cloud-based data access and sharing software": "Complete cloud-based data access and file-sharing tools training.",
    "Cloud-based management software": "Complete cloud management platform training (AWS Console, Azure Portal).",
    "Cloud-based protection or security software": "Complete cloud security and protection tools training.",
    "Communications server software":  "Complete communications server and messaging platform training.",
    "Desktop communications software": "Complete desktop communication tools training (Teams, Slack, Zoom).",
    "Device drivers or system software": "Complete system-level and device driver software training.",
    "Filesystem software":             "Complete file system administration and storage management training.",
    "Gateway software":                "Complete network gateway and routing software training.",
    "Internet directory services software": "Complete directory services and LDAP management training.",
    "Internet protocol IP multimedia subsystem software": "Complete IP multimedia subsystem and VoIP training.",
    "LAN software":                    "Complete LAN management and network administration training.",
    "Network connectivity terminal emulation software": "Complete network terminal emulation tools training.",
    "Network conferencing software":   "Complete network conferencing and virtual meeting tools training.",
    "Network monitoring software":     "Complete network monitoring and performance analysis tools training.",
    "Network operating system enhancement software": "Complete network OS enhancement and optimization training.",
    "Network operation system software": "Complete network operating system administration training.",
    "Network security and virtual private network VPN equipment software": "Complete VPN and network security equipment training.",
    "Network security or virtual private network VPN management software": "Complete VPN management and network security administration training.",
    "Platform interconnectivity software": "Complete platform integration and API management training.",
    "Portal server software":          "Complete enterprise portal server and content management training.",
    "Software defined networking/ virtualization software": "Complete SDN and network virtualization training.",
    "Storage media loading software":  "Complete storage media and hardware operations training.",
    "Storage networking software":     "Complete SAN and storage networking infrastructure training.",
    "Switch or router software":       "Complete switch and router configuration and management training.",
    "WAN switching software and firmware": "Complete WAN switching and network firmware management training.",
    "Wireless software":               "Complete wireless network administration and Wi-Fi management training.",

    # ── Industry / Domain Software ───────────────────────────────────────────
    "Aviation ground support software":     "Complete aviation ground support operations software training.",
    "Computer aided design CAD and computer aided manufacturing CAM system": "Complete CAD/CAM system training (AutoCAD, SolidWorks).",
    "Computer aided design CAD software":   "Complete CAD software training (AutoCAD, Fusion 360).",
    "Computer aided manufacturing CAM software": "Complete CAM software training for manufacturing automation.",
    "Computer based training software":     "Complete e-learning authoring and CBT platform training.",
    "Computer imaging software":            "Complete computer imaging and medical imaging software training.",
    "Content workflow software":            "Complete content workflow and editorial management tools training.",
    "Customer relationship management CRM software": "Complete CRM platform training (Salesforce, HubSpot, Dynamics).",
    "Enterprise application integration software": "Complete enterprise integration and ESB tools training.",
    "Enterprise resource planning ERP software": "Complete ERP platform training (SAP, Oracle, NetSuite).",
    "Enterprise system management software": "Complete enterprise systems management and ITSM tools training.",
    "Expert system software":               "Complete expert system and AI decision-support tools training.",
    "Eye tracking software":                "Complete eye tracking and usability analysis tools training.",
    "Facilities management software":       "Complete facilities management and maintenance scheduling software training.",
    "Flight control software":              "Complete flight control and aviation systems software training.",
    "Graphical user interface development software": "Complete GUI development tools and frameworks training.",
    "Human resources software":             "Complete HRIS and human resources management software training.",
    "Industrial control software":          "Complete industrial control and SCADA systems training.",
    "Information retrieval or search software": "Complete information retrieval and enterprise search tools training.",
    "Interactive voice response software":  "Complete IVR and voice response system training.",
    "Manufacturing execution system MES software": "Complete MES and manufacturing operations software training.",
    "Pattern design software":              "Complete pattern design and CAD drafting tools training.",
    "Point of sale POS software":           "Complete POS system and retail operations software training.",
    "Procedure management software":        "Complete procedure and compliance management software training.",
    "Sales and marketing software":         "Complete sales and marketing automation tools training.",
    "Transaction security and virus protection software": "Complete transaction security and antivirus tools training.",
    "Transaction server software":          "Complete transaction server and middleware administration training.",
    "Voice recognition software":           "Complete voice recognition and speech-to-text tools training.",
    "Voice synthesizer and recognition software": "Complete voice synthesis and recognition technology training.",
    "Action games":                         "Complete interactive simulation and game-based learning tools training.",
}


# Fallback that still references the skill name (requirement 12)
def _default_recommendation(skill: str) -> str:
    return f"Consider targeted training or certification in {skill}."


def recommend_for_skill(skill: str) -> str:
    """Map a missing skill to a specific training recommendation."""
    return SKILL_TO_RECOMMENDATION.get(skill, _default_recommendation(skill))


def generate_recommendations(
    employee_roles: list[dict],
    employee_skills: dict[str, set],
) -> list[dict]:
    """
    Generate upskilling recommendations for each employee.

    Returns:
        [{"employee_id": "101", "role": "Data Analyst",
          "missing_skills": ["Critical Thinking", ...],
          "recommendations": ["Enroll in critical thinking...", ...]}, ...]
    """
    gaps = compute_employee_skill_gaps(employee_roles, employee_skills)

    results = []
    for emp in gaps:
        seen = set()
        recommendations = []
        for skill in emp["missing_skills"]:
            rec = recommend_for_skill(skill)
            # Deduplicate: if same skill maps to same rec, skip (requirement 9)
            key = (skill, rec)
            if key not in seen:
                seen.add(key)
                recommendations.append(f"{skill} -> {rec}")

        results.append({
            "employee_id": emp["employee_id"],
            "role": emp["role"],
            "gap_count": emp["gap_count"],
            "missing_skills": emp["missing_skills"],
            "recommendations": recommendations,
        })

    return results
