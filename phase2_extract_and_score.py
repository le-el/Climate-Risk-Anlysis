"""
Phase 2-4: Extract structured data using GPT-4o-mini and score using rubric rules
"""

import os
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import re
from tqdm import tqdm

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_framework(framework_path):
    """Load the framework Excel file"""
    df = pd.read_excel(framework_path, sheet_name='PhysicalRisk_Resilience_Framewo')
    return df

def load_company_chunks(company_id, chunks_folder="preprocessed_chunks"):
    """Load preprocessed chunks for a company"""
    chunks_file = os.path.join(chunks_folder, f"{company_id}_chunks.json")
    if not os.path.exists(chunks_file):
        return []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_keywords(keyword_string):
    """Extract English keywords from keyword string"""
    if pd.isna(keyword_string):
        return []
    
    keywords_str = str(keyword_string)
    en_match = re.search(r'EN:\s*([^|]+)', keywords_str, re.IGNORECASE)
    if en_match:
        en_section = en_match.group(1).strip()
        keywords = [kw.strip().lower() for kw in en_section.split(';') if kw.strip()]
        return list(set([kw for kw in keywords if len(kw) > 2]))
    return []

def find_relevant_chunks(chunks, keywords, top_n=6):
    """Find top chunks containing the keywords"""
    scored_chunks = []
    
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            # Try exact word boundary match first
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                score += 2
                matched_keywords.append(keyword)
            # Also try phrase match
            elif keyword in text_lower:
                score += 1
                matched_keywords.append(keyword)
        
        if score > 0:
            scored_chunks.append({
                **chunk,
                "score": score,
                "matched_keywords": matched_keywords
            })
    
    # Sort by score and return top N
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_n]

def get_json_schema_for_measure(measure_name, definition):
    """Generate JSON schema based on measure"""
    
    # Define measure-specific schemas
    schemas = {
        "Board-level physical risk oversight": {
            "committee_name": "string",
            "charter_url": "string",
            "charter_last_updated_date": "string",
            "review_frequency_per_year": "number",
            "hazards_count": "number",
            "horizons": "array",
            "subsidiaries_coverage_pct": "number"
        },
        "Senior management responsibility": {
            "executive_name": "string",
            "executive_title": "string",
            "reporting_line": "string",
            "team_size_ftes": "number",
            "has_budget": "boolean",
            "has_kpis": "boolean",
            "reporting_frequency_per_year": "number",
            "scope_coverage_pct": "number"
        },
        "ERM integration of physical risk": {
            "in_risk_taxonomy": "boolean",
            "has_risk_appetite": "boolean",
            "has_tolerances": "boolean",
            "in_risk_register": "boolean",
            "linked_to_strategy": "boolean",
            "linked_to_capex": "boolean",
            "update_frequency_per_year": "number",
            "functions_covered": "array",
            "multi_hazard_analysis": "boolean"
        },
        "Physical risk scenario analysis": {
            "scenarios_count": "number",
            "scenario_types": "array",
            "horizons_analyzed": "array",
            "assets_coverage_pct": "number",
            "hazards_analyzed": "array",
            "vendor_models_used": "array",
            "return_periods_included": "boolean",
            "frequency_per_year": "number"
        },
        "Geographic risk mapping": {
            "assets_geocoded_pct": "number",
            "hazards_mapped": "array",
            "update_frequency_per_year": "number",
            "has_gis_system": "boolean",
            "horizons_mapped": "array",
            "vendor_used": "string",
            "validation_done": "boolean"
        },
        "Business continuity planning coverage": {
            "coverage_pct": "number",
            "facilities_covered": "number",
            "has_bcp_document": "boolean",
            "tested_frequency_per_year": "number",
            "includes_physical_risk": "boolean",
            "multi_hazard_coverage": "boolean"
        },
        "Insurance program structure": {
            "has_property_insurance": "boolean",
            "has_business_interruption": "boolean",
            "coverage_amount": "number",
            "deductible_amount": "number",
            "insurer_names": "array",
            "renewal_frequency": "string",
            "includes_climate_risk": "boolean"
        },
        "Financial quantification disclosure": {
            "exposure_by_peril": "object",
            "exposure_by_region": "object",
            "insured_vs_uninsured": "object",
            "business_interruption_included": "boolean",
            "horizons_disclosed": "array",
            "confidence_intervals": "boolean",
            "aal_disclosed": "boolean",
            "pml_disclosed": "boolean"
        },
        "Public adaptation/resilience commitments": {
            "has_targets": "boolean",
            "target_description": "string",
            "budget_amount": "number",
            "capex_allocated": "number",
            "time_bound": "boolean",
            "target_date": "string",
            "sites_coverage_pct": "number",
            "multi_hazard": "boolean"
        },
        "Incentives linked to physical risk KPIs": {
            "has_kpi_linkage": "boolean",
            "kpi_description": "string",
            "compensation_weight_pct": "number",
            "executives_covered_pct": "number",
            "kpis_are_measurable": "boolean",
            "has_audit": "boolean"
        },
        "Climate competency and training": {
            "training_programs_count": "number",
            "employees_trained_pct": "number",
            "executives_trained_pct": "number",
            "training_frequency_per_year": "number",
            "training_topics": "array",
            "certification_programs": "array",
            "has_budget": "boolean"
        },
        "Disclosure alignment": {
            "frameworks_used": "array",
            "alignment_score_pct": "number",
            "metrics_disclosed_count": "number",
            "assurance_provided": "boolean",
            "reporting_frequency": "string",
            "third_party_verification": "boolean"
        },
        "Sector/portfolio risk assessment": {
            "sectors_assessed_count": "number",
            "assessment_frequency_per_year": "number",
            "risk_levels_identified": "array",
            "mitigation_plans_count": "number",
            "hazards_assessed": "array",
            "horizons_analyzed": "array"
        },
        "Stress testing integration": {
            "stress_tests_count": "number",
            "includes_physical_risk": "boolean",
            "test_frequency_per_year": "number",
            "scenarios_tested": "array",
            "results_disclosed": "boolean",
            "regulatory_compliance": "boolean"
        },
        "Financial quantification methodology": {
            "methodology_documented": "boolean",
            "valuation_approach": "string",
            "models_used": "array",
            "assumptions_disclosed": "boolean",
            "validation_done": "boolean",
            "uncertainty_addressed": "boolean"
        },
        "Dependency mapping": {
            "dependencies_mapped_count": "number",
            "critical_dependencies_count": "number",
            "update_frequency_per_year": "number",
            "includes_external_deps": "boolean",
            "risk_assessments_done": "boolean",
            "mitigation_plans_count": "number"
        },
        "Tail-risk treatment": {
            "tail_risks_identified_count": "number",
            "treatment_methods": "array",
            "stress_scenarios_count": "number",
            "capital_allocation_pct": "number",
            "monitoring_in_place": "boolean",
            "board_review_frequency": "number"
        },
        "Data quality & validation": {
            "validation_procedures_count": "number",
            "data_sources_count": "number",
            "third_party_validation": "boolean",
            "accuracy_checks_done": "boolean",
            "update_frequency_per_year": "number",
            "controls_documented": "boolean"
        },
        "Facility design standards": {
            "standards_documented": "boolean",
            "standards_applied_pct": "number",
            "new_facilities_compliant_pct": "number",
            "hazards_covered": "array",
            "review_frequency_per_year": "number",
            "third_party_certification": "boolean"
        },
        "Infrastructure adaptation measures": {
            "measures_implemented_count": "number",
            "investment_amount": "number",
            "sites_coverage_pct": "number",
            "hazards_addressed": "array",
            "measures_planned_count": "number",
            "effectiveness_monitored": "boolean"
        },
        "Technology system resilience": {
            "systems_assessed_count": "number",
            "resilience_measures_count": "number",
            "backup_systems_pct": "number",
            "disaster_recovery_tested": "boolean",
            "downtime_target_hours": "number",
            "cyber_physical_risks_covered": "boolean"
        },
        "Location risk policy": {
            "policy_documented": "boolean",
            "prohibited_zones_defined": "boolean",
            "assessment_required": "boolean",
            "new_locations_assessed_pct": "number",
            "review_frequency_per_year": "number",
            "board_approved": "boolean"
        },
        "Redundancy and autonomy": {
            "redundant_systems_count": "number",
            "autonomous_systems_pct": "number",
            "critical_functions_covered_pct": "number",
            "testing_frequency_per_year": "number",
            "recovery_time_target_hours": "number",
            "geographic_diversification": "boolean"
        },
        "Emergency response procedures": {
            "procedures_documented": "boolean",
            "facilities_covered_pct": "number",
            "drills_conducted_per_year": "number",
            "response_time_target_minutes": "number",
            "team_trained_pct": "number",
            "equipment_available": "boolean"
        },
        "Communication protocols": {
            "protocols_documented": "boolean",
            "stakeholders_covered": "array",
            "channels_established_count": "number",
            "tested_frequency_per_year": "number",
            "escalation_paths_defined": "boolean",
            "crisis_communication_plan": "boolean"
        },
        "Early warning systems": {
            "systems_deployed_count": "number",
            "hazards_monitored": "array",
            "coverage_pct": "number",
            "alert_time_hours": "number",
            "integration_with_operations": "boolean",
            "tested_frequency_per_year": "number"
        },
        "Incident reviews & remediation": {
            "incidents_reviewed_count": "number",
            "review_frequency": "string",
            "remediation_plans_count": "number",
            "lessons_learned_documented": "boolean",
            "board_reported": "boolean",
            "preventive_measures_count": "number"
        },
        "Supplier physical risk assessment": {
            "suppliers_assessed_pct": "number",
            "assessment_frequency_per_year": "number",
            "critical_suppliers_coverage_pct": "number",
            "risk_levels_identified": "array",
            "mitigation_required": "boolean",
            "results_tracked": "boolean"
        },
        "Contractual resilience requirements": {
            "suppliers_covered_pct": "number",
            "requirements_documented": "boolean",
            "contracts_reviewed_pct": "number",
            "compliance_monitored": "boolean",
            "enforcement_mechanisms": "array",
            "critical_suppliers_pct": "number"
        },
        "Monitoring & audits": {
            "audits_conducted_per_year": "number",
            "suppliers_audited_pct": "number",
            "third_party_audits": "boolean",
            "non_compliance_corrected_pct": "number",
            "audit_results_reported": "boolean",
            "continuous_monitoring": "boolean"
        },
        "Alternative sourcing & buffers": {
            "alternatives_identified_count": "number",
            "buffer_inventory_days": "number",
            "critical_suppliers_diversified_pct": "number",
            "switching_capability_tested": "boolean",
            "geographic_diversification": "boolean",
            "suppliers_qualified_count": "number"
        },
        "Supplier disclosure requirements": {
            "suppliers_required_pct": "number",
            "disclosure_frameworks": "array",
            "compliance_verified": "boolean",
            "enforcement_actions_count": "number",
            "disclosures_reviewed_pct": "number",
            "third_party_validation": "boolean"
        },
        "Business interruption coverage": {
            "has_business_interruption": "boolean",
            "coverage_amount": "number",
            "waiting_period_days": "number",
            "indemnity_period_days": "number",
            "coverage_extent_pct": "number",
            "premium_amount": "number"
        },
        "Captives/parametric solutions": {
            "has_captive": "boolean",
            "has_parametric": "boolean",
            "coverage_amount": "number",
            "triggers_defined": "array",
            "payout_structure": "string",
            "capitalization_amount": "number"
        },
        "Insurance gap": {
            "gap_calculated": "boolean",
            "total_gap_amount": "number",
            "gap_by_hazard": "object",
            "mitigation_plans_count": "number",
            "board_reviewed": "boolean",
            "coverage_target_pct": "number"
        },
        "Internal controls over climate data": {
            "controls_documented": "boolean",
            "controls_tested_frequency": "number",
            "deficiencies_remediated_pct": "number",
            "management_review": "boolean",
            "data_governance_framework": "boolean",
            "automated_controls_count": "number"
        },
        "External assurance": {
            "assurance_provided": "boolean",
            "assurance_level": "string",
            "assurer_name": "string",
            "scope_coverage_pct": "number",
            "assurance_frequency": "string",
            "publicly_disclosed": "boolean"
        },
        "Workforce heat/AQI thresholds & protocols": {
            "thresholds_defined": "boolean",
            "protocols_documented": "boolean",
            "sites_covered_pct": "number",
            "workforce_covered_pct": "number",
            "monitoring_systems_count": "number",
            "compliance_verified": "boolean"
        },
        "Water security/drought controls": {
            "controls_implemented_count": "number",
            "water_sources_diversified_count": "number",
            "conservation_target_pct": "number",
            "sites_covered_pct": "number",
            "monitoring_in_place": "boolean",
            "contingency_plans_count": "number"
        },
        "Community/utility engagement": {
            "engagement_programs_count": "number",
            "partnerships_count": "number",
            "stakeholders_engaged": "array",
            "initiatives_funded": "number",
            "resilience_projects_count": "number",
            "engagement_frequency": "string"
        },
        "Downtime and service impact": {
            "downtime_tracked": "boolean",
            "target_downtime_hours": "number",
            "actual_downtime_hours": "number",
            "service_impact_measured": "boolean",
            "customer_affected_count": "number",
            "recovery_time_hours": "number"
        },
        "Losses and loss ratios": {
            "losses_tracked": "boolean",
            "total_losses_amount": "number",
            "insured_losses_amount": "number",
            "loss_ratio_pct": "number",
            "by_hazard": "object",
            "trend_analysis_done": "boolean"
        },
        "Supplier disruption days": {
            "disruptions_tracked": "boolean",
            "total_disruption_days": "number",
            "critical_supplier_disruptions": "number",
            "average_disruption_days": "number",
            "by_supplier_category": "object",
            "mitigation_effectiveness": "number"
        },
        "Adaptation spend and outcomes": {
            "spend_tracked": "boolean",
            "total_spend_amount": "number",
            "capex_spend_pct": "number",
            "outcomes_measured": "boolean",
            "effectiveness_metrics": "array",
            "roi_calculated": "boolean"
        }
    }
    
    # Return schema for specific measure or generic
    if measure_name in schemas:
        return schemas[measure_name]
    else:
        # Generic schema (shouldn't be reached if all measures are implemented)
        return {
            "has_mention": "boolean",
            "description": "string",
            "coverage_pct": "number",
            "frequency": "string"
        }

def extract_with_llm(measure_name, definition, relevant_chunks, keywords):
    """Extract structured data using GPT-4o-mini"""
    
    # Get schema for this measure
    schema = get_json_schema_for_measure(measure_name, definition)
    
    # Prepare context text
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"[Document: {chunk['doc_id']}, Page: {chunk['page']}]\n{chunk['text']}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Build field descriptions from schema
    field_descriptions = []
    for field_name, field_type in schema.items():
        desc = f"- {field_name} ({field_type}): "
        if field_type == "number":
            desc += "Numeric value (use 0 if not stated)"
        elif field_type == "boolean":
            desc += "true or false"
        elif field_type == "array":
            desc += "Array of strings or objects"
        elif field_type == "object":
            desc += "Object with nested properties"
        else:
            desc += "String value (leave blank if not stated)"
        field_descriptions.append(desc)
    
    fields_text = "\n".join(field_descriptions)
    
    # Build prompt
    prompt = f"""You are extracting structured data for the measure: "{measure_name}"

Definition: {definition}

Instructions:
1. Extract ONLY information explicitly stated in the text below
2. Do NOT guess or infer values
3. Leave fields blank, 0, false, or empty array if not explicitly stated
4. For frequencies: "semi-annually" = 2, "quarterly" = 4, "annually" = 1, "monthly" = 12
5. For percentages: extract as number (0-100)
6. Always include an "evidence" object with doc_id, page, and the exact snippet

Return a JSON object with these fields:
{fields_text}
- evidence: Object with doc_id (string), page (number), and snippet (string) where information was found

Context text:
{context}

Return ONLY valid JSON, no additional text:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return {}

def score_board_oversight(fields):
    """Score board oversight measure using rubric (0-5 scale)"""
    cov = fields.get("subsidiaries_coverage_pct", 0)
    freq = fields.get("review_frequency_per_year", 0)
    hazards = fields.get("hazards_count", 0)
    horizons = set(fields.get("horizons", []))
    committee_name = fields.get("committee_name", "")
    
    # 0: No board reference
    if not committee_name and hazards == 0:
        return 0
    
    # 1: Mentions climate oversight without physical risk specificity
    # (Simplified - assume if we found something, it's at least level 1)
    if committee_name or hazards > 0:
        score = 1
    else:
        return 0
    
    # 2: Names a board/committee but physical risk not explicit; or ad hoc review; <50% coverage
    if committee_name and (freq < 1 or cov < 50):
        score = 2
    
    # 3: Explicit oversight, ≥ annual, ≥50% coverage, hazards or horizons
    if freq >= 1 and cov >= 50 and (hazards >= 1 or len(horizons) > 0):
        score = 3
    
    # 4: Charter covers acute+chronic hazards, ≥2x/year, ≥80% coverage, Near/Medium/Long
    if freq >= 2 and cov >= 80 and hazards >= 2 and {"Near", "Medium", "Long"}.issubset(horizons):
        score = 4
    
    # 5: ≥ quarterly, ≥95% coverage
    if freq >= 4 and cov >= 95:
        score = 5
    
    return score

def score_senior_management(fields):
    """Score senior management responsibility"""
    exec_name = fields.get("executive_name", "")
    has_budget = fields.get("has_budget", False)
    has_kpis = fields.get("has_kpis", False)
    freq = fields.get("reporting_frequency_per_year", 0)
    scope = fields.get("scope_coverage_pct", 0)
    
    if not exec_name:
        return 0
    
    score = 1
    if exec_name and fields.get("reporting_line", ""):
        score = 2
    
    if exec_name and freq >= 1 and scope >= 50:
        score = 3
    
    if exec_name and has_budget and has_kpis and freq >= 2 and scope >= 80:
        score = 4
    
    if exec_name and has_budget and has_kpis and freq >= 2 and scope >= 95 and fields.get("team_size_ftes", 0) > 0:
        score = 5
    
    return score

def score_erm_integration(fields):
    """Score ERM integration of physical risk"""
    in_taxonomy = fields.get("in_risk_taxonomy", False)
    has_appetite = fields.get("has_risk_appetite", False)
    has_tolerances = fields.get("has_tolerances", False)
    linked_strategy = fields.get("linked_to_strategy", False)
    linked_capex = fields.get("linked_to_capex", False)
    functions = len(fields.get("functions_covered", []))
    
    if not in_taxonomy and not has_appetite:
        return 0
    
    score = 1
    if in_taxonomy or has_appetite:
        score = 2
    
    if in_taxonomy and has_tolerances and functions >= 1:
        score = 3
    
    if in_taxonomy and has_tolerances and linked_strategy and functions >= 3 and fields.get("multi_hazard_analysis", False):
        score = 4
    
    if in_taxonomy and has_tolerances and linked_strategy and linked_capex and functions >= 5:
        score = 5
    
    return score

def score_scenario_analysis(fields):
    """Score physical risk scenario analysis"""
    scenarios = fields.get("scenarios_count", 0)
    assets_cov = fields.get("assets_coverage_pct", 0)
    hazards = len(fields.get("hazards_analyzed", []))
    horizons = len(fields.get("horizons_analyzed", []))
    
    if scenarios == 0:
        return 0
    
    score = 1
    if scenarios >= 1:
        score = 2
    
    if scenarios >= 2 and assets_cov >= 50 and hazards >= 1:
        score = 3
    
    if scenarios >= 3 and assets_cov >= 80 and hazards >= 2 and horizons >= 2:
        score = 4
    
    if scenarios >= 3 and assets_cov >= 95 and hazards >= 3 and horizons >= 3 and fields.get("return_periods_included", False):
        score = 5
    
    return score

def score_geographic_mapping(fields):
    """Score geographic risk mapping"""
    assets_geo = fields.get("assets_geocoded_pct", 0)
    hazards_mapped = len(fields.get("hazards_mapped", []))
    has_gis = fields.get("has_gis_system", False)
    validation = fields.get("validation_done", False)
    
    if assets_geo == 0:
        return 0
    
    score = 1
    if assets_geo < 50 or hazards_mapped < 1:
        score = 2
    
    if assets_geo >= 50 and hazards_mapped >= 3 and fields.get("update_frequency_per_year", 0) >= 1:
        score = 3
    
    if assets_geo >= 80 and hazards_mapped >= 5 and has_gis and len(fields.get("horizons_mapped", [])) >= 2:
        score = 4
    
    if assets_geo >= 95 and hazards_mapped >= 5 and has_gis and validation:
        score = 5
    
    return score

def score_bcp_coverage(fields):
    """Score business continuity planning coverage"""
    coverage = fields.get("coverage_pct", 0)
    has_doc = fields.get("has_bcp_document", False)
    includes_physical = fields.get("includes_physical_risk", False)
    tested_freq = fields.get("tested_frequency_per_year", 0)
    
    if coverage == 0 and not has_doc:
        return 0
    
    score = 1
    if has_doc:
        score = 2
    
    if has_doc and coverage >= 50 and includes_physical:
        score = 3
    
    if has_doc and coverage >= 80 and includes_physical and tested_freq >= 1 and fields.get("multi_hazard_coverage", False):
        score = 4
    
    if has_doc and coverage >= 95 and includes_physical and tested_freq >= 1 and fields.get("multi_hazard_coverage", False):
        score = 5
    
    return score

def score_insurance_structure(fields):
    """Score insurance program structure"""
    has_property = fields.get("has_property_insurance", False)
    has_bi = fields.get("has_business_interruption", False)
    includes_climate = fields.get("includes_climate_risk", False)
    
    if not has_property:
        return 0
    
    score = 1
    if has_property:
        score = 2
    
    if has_property and has_bi:
        score = 3
    
    if has_property and has_bi and includes_climate:
        score = 4
    
    if has_property and has_bi and includes_climate and fields.get("coverage_amount", 0) > 0:
        score = 5
    
    return score

def score_financial_disclosure(fields):
    """Score financial quantification disclosure"""
    exposure_peril = fields.get("exposure_by_peril", {})
    exposure_region = fields.get("exposure_by_region", {})
    insured_uninsured = fields.get("insured_vs_uninsured", {})
    has_bi = fields.get("business_interruption_included", False)
    horizons = len(fields.get("horizons_disclosed", []))
    
    if not exposure_peril and not exposure_region:
        return 0
    
    score = 1
    if exposure_peril or exposure_region:
        score = 2
    
    if (exposure_peril or exposure_region) and horizons >= 1:
        score = 3
    
    if exposure_peril and exposure_region and insured_uninsured and horizons >= 2:
        score = 4
    
    if exposure_peril and exposure_region and insured_uninsured and has_bi and horizons >= 3 and (fields.get("aal_disclosed", False) or fields.get("pml_disclosed", False)):
        score = 5
    
    return score

def score_adaptation_commitments(fields):
    """Score public adaptation/resilience commitments"""
    has_targets = fields.get("has_targets", False)
    budget = fields.get("budget_amount", 0)
    capex = fields.get("capex_allocated", 0)
    time_bound = fields.get("time_bound", False)
    sites_cov = fields.get("sites_coverage_pct", 0)
    multi_hazard = fields.get("multi_hazard", False)
    
    if not has_targets:
        return 0
    
    score = 1
    if has_targets:
        score = 2
    
    if has_targets and time_bound and (budget > 0 or capex > 0) and sites_cov >= 50:
        score = 3
    
    if has_targets and time_bound and (budget > 0 or capex > 0) and sites_cov >= 80 and multi_hazard:
        score = 4
    
    if has_targets and time_bound and (budget > 0 or capex > 0) and sites_cov >= 95 and multi_hazard:
        score = 5
    
    return score

def score_kpi_incentives(fields):
    """Score incentives linked to physical risk KPIs"""
    has_linkage = fields.get("has_kpi_linkage", False)
    weight = fields.get("compensation_weight_pct", 0)
    exec_cov = fields.get("executives_covered_pct", 0)
    measurable = fields.get("kpis_are_measurable", False)
    has_audit = fields.get("has_audit", False)
    
    if not has_linkage:
        return 0
    
    score = 1
    if has_linkage and not measurable:
        score = 2
    
    if has_linkage and measurable and weight >= 10 and exec_cov >= 50:
        score = 3
    
    if has_linkage and measurable and weight >= 10 and exec_cov >= 80:
        score = 4
    
    if has_linkage and measurable and weight >= 10 and exec_cov >= 95 and has_audit:
        score = 5
    
    return score

def score_climate_competency(fields):
    """Score climate competency and training"""
    programs = fields.get("training_programs_count", 0)
    employees = fields.get("employees_trained_pct", 0)
    execs = fields.get("executives_trained_pct", 0)
    freq = fields.get("training_frequency_per_year", 0)
    
    if programs == 0:
        return 0
    
    score = 1
    if programs >= 1:
        score = 2
    
    if programs >= 1 and employees >= 50 and freq >= 1:
        score = 3
    
    if programs >= 2 and employees >= 80 and execs >= 50 and freq >= 1:
        score = 4
    
    if programs >= 2 and employees >= 95 and execs >= 95 and freq >= 2 and fields.get("has_budget", False):
        score = 5
    
    return score

def score_disclosure_alignment(fields):
    """Score disclosure alignment"""
    frameworks = len(fields.get("frameworks_used", []))
    alignment = fields.get("alignment_score_pct", 0)
    metrics = fields.get("metrics_disclosed_count", 0)
    assurance = fields.get("assurance_provided", False)
    
    if frameworks == 0:
        return 0
    
    score = 1
    if frameworks >= 1:
        score = 2
    
    if frameworks >= 1 and alignment >= 50 and metrics >= 5:
        score = 3
    
    if frameworks >= 2 and alignment >= 80 and metrics >= 10 and assurance:
        score = 4
    
    if frameworks >= 2 and alignment >= 95 and metrics >= 15 and assurance and fields.get("third_party_verification", False):
        score = 5
    
    return score

def score_sector_assessment(fields):
    """Score sector/portfolio risk assessment"""
    sectors = fields.get("sectors_assessed_count", 0)
    freq = fields.get("assessment_frequency_per_year", 0)
    hazards = len(fields.get("hazards_assessed", []))
    
    if sectors == 0:
        return 0
    
    score = 1
    if sectors >= 1:
        score = 2
    
    if sectors >= 2 and freq >= 1 and hazards >= 1:
        score = 3
    
    if sectors >= 3 and freq >= 1 and hazards >= 3 and fields.get("mitigation_plans_count", 0) >= 1:
        score = 4
    
    if sectors >= 3 and freq >= 2 and hazards >= 5 and fields.get("mitigation_plans_count", 0) >= 3:
        score = 5
    
    return score

def score_stress_testing(fields):
    """Score stress testing integration"""
    tests = fields.get("stress_tests_count", 0)
    includes_pr = fields.get("includes_physical_risk", False)
    freq = fields.get("test_frequency_per_year", 0)
    scenarios = len(fields.get("scenarios_tested", []))
    
    if not includes_pr or tests == 0:
        return 0
    
    score = 1
    if includes_pr and tests >= 1:
        score = 2
    
    if includes_pr and tests >= 1 and freq >= 1 and scenarios >= 1:
        score = 3
    
    if includes_pr and tests >= 2 and freq >= 1 and scenarios >= 2 and fields.get("results_disclosed", False):
        score = 4
    
    if includes_pr and tests >= 2 and freq >= 2 and scenarios >= 3 and fields.get("results_disclosed", False) and fields.get("regulatory_compliance", False):
        score = 5
    
    return score

def score_quant_methodology(fields):
    """Score financial quantification methodology"""
    documented = fields.get("methodology_documented", False)
    validated = fields.get("validation_done", False)
    assumptions = fields.get("assumptions_disclosed", False)
    uncertainty = fields.get("uncertainty_addressed", False)
    
    if not documented:
        return 0
    
    score = 1
    if documented:
        score = 2
    
    if documented and validated:
        score = 3
    
    if documented and validated and assumptions and uncertainty:
        score = 4
    
    if documented and validated and assumptions and uncertainty and fields.get("models_used", []) and len(fields.get("models_used", [])) >= 1:
        score = 5
    
    return score

def score_dependency_mapping(fields):
    """Score dependency mapping"""
    deps = fields.get("dependencies_mapped_count", 0)
    critical = fields.get("critical_dependencies_count", 0)
    freq = fields.get("update_frequency_per_year", 0)
    
    if deps == 0:
        return 0
    
    score = 1
    if deps >= 1:
        score = 2
    
    if deps >= 5 and freq >= 1 and fields.get("risk_assessments_done", False):
        score = 3
    
    if deps >= 10 and critical >= 3 and freq >= 1 and fields.get("includes_external_deps", False):
        score = 4
    
    if deps >= 20 and critical >= 5 and freq >= 2 and fields.get("mitigation_plans_count", 0) >= 3:
        score = 5
    
    return score

def score_tail_risk(fields):
    """Score tail-risk treatment"""
    risks = fields.get("tail_risks_identified_count", 0)
    methods = len(fields.get("treatment_methods", []))
    scenarios = fields.get("stress_scenarios_count", 0)
    
    if risks == 0:
        return 0
    
    score = 1
    if risks >= 1:
        score = 2
    
    if risks >= 1 and methods >= 1 and fields.get("monitoring_in_place", False):
        score = 3
    
    if risks >= 2 and methods >= 2 and scenarios >= 1 and fields.get("board_review_frequency", 0) >= 1:
        score = 4
    
    if risks >= 3 and methods >= 3 and scenarios >= 2 and fields.get("capital_allocation_pct", 0) > 0 and fields.get("board_review_frequency", 0) >= 1:
        score = 5
    
    return score

def score_data_quality(fields):
    """Score data quality & validation"""
    procedures = fields.get("validation_procedures_count", 0)
    sources = fields.get("data_sources_count", 0)
    third_party = fields.get("third_party_validation", False)
    documented = fields.get("controls_documented", False)
    
    if procedures == 0:
        return 0
    
    score = 1
    if procedures >= 1:
        score = 2
    
    if procedures >= 2 and documented and fields.get("accuracy_checks_done", False):
        score = 3
    
    if procedures >= 3 and sources >= 3 and documented and third_party:
        score = 4
    
    if procedures >= 5 and sources >= 5 and documented and third_party and fields.get("update_frequency_per_year", 0) >= 2:
        score = 5
    
    return score

def score_facility_design(fields):
    """Score facility design standards"""
    documented = fields.get("standards_documented", False)
    applied = fields.get("standards_applied_pct", 0)
    compliant = fields.get("new_facilities_compliant_pct", 0)
    hazards = len(fields.get("hazards_covered", []))
    
    if not documented:
        return 0
    
    score = 1
    if documented:
        score = 2
    
    if documented and applied >= 50 and compliant >= 50:
        score = 3
    
    if documented and applied >= 80 and compliant >= 80 and hazards >= 3:
        score = 4
    
    if documented and applied >= 95 and compliant >= 95 and hazards >= 5 and fields.get("third_party_certification", False):
        score = 5
    
    return score

def score_infrastructure_adaptation(fields):
    """Score infrastructure adaptation measures"""
    measures = fields.get("measures_implemented_count", 0)
    coverage = fields.get("sites_coverage_pct", 0)
    hazards = len(fields.get("hazards_addressed", []))
    investment = fields.get("investment_amount", 0)
    
    if measures == 0:
        return 0
    
    score = 1
    if measures >= 1:
        score = 2
    
    if measures >= 2 and coverage >= 50 and hazards >= 1:
        score = 3
    
    if measures >= 3 and coverage >= 80 and hazards >= 3 and investment > 0:
        score = 4
    
    if measures >= 5 and coverage >= 95 and hazards >= 5 and investment > 0 and fields.get("effectiveness_monitored", False):
        score = 5
    
    return score

def score_technology_resilience(fields):
    """Score technology system resilience"""
    assessed = fields.get("systems_assessed_count", 0)
    measures = fields.get("resilience_measures_count", 0)
    backup = fields.get("backup_systems_pct", 0)
    tested = fields.get("disaster_recovery_tested", False)
    
    if assessed == 0:
        return 0
    
    score = 1
    if assessed >= 1:
        score = 2
    
    if assessed >= 3 and backup >= 50 and tested:
        score = 3
    
    if assessed >= 5 and backup >= 80 and tested and measures >= 3:
        score = 4
    
    if assessed >= 10 and backup >= 95 and tested and measures >= 5 and fields.get("cyber_physical_risks_covered", False):
        score = 5
    
    return score

def score_location_policy(fields):
    """Score location risk policy"""
    documented = fields.get("policy_documented", False)
    assessment = fields.get("assessment_required", False)
    coverage = fields.get("new_locations_assessed_pct", 0)
    
    if not documented:
        return 0
    
    score = 1
    if documented:
        score = 2
    
    if documented and assessment and coverage >= 50:
        score = 3
    
    if documented and assessment and coverage >= 80 and fields.get("prohibited_zones_defined", False):
        score = 4
    
    if documented and assessment and coverage >= 95 and fields.get("prohibited_zones_defined", False) and fields.get("board_approved", False):
        score = 5
    
    return score

def score_redundancy(fields):
    """Score redundancy and autonomy"""
    redundant = fields.get("redundant_systems_count", 0)
    autonomous = fields.get("autonomous_systems_pct", 0)
    coverage = fields.get("critical_functions_covered_pct", 0)
    
    if redundant == 0:
        return 0
    
    score = 1
    if redundant >= 1:
        score = 2
    
    if redundant >= 2 and coverage >= 50:
        score = 3
    
    if redundant >= 3 and coverage >= 80 and autonomous >= 30:
        score = 4
    
    if redundant >= 5 and coverage >= 95 and autonomous >= 50 and fields.get("geographic_diversification", False):
        score = 5
    
    return score

def score_emergency_response(fields):
    """Score emergency response procedures"""
    documented = fields.get("procedures_documented", False)
    coverage = fields.get("facilities_covered_pct", 0)
    drills = fields.get("drills_conducted_per_year", 0)
    trained = fields.get("team_trained_pct", 0)
    
    if not documented:
        return 0
    
    score = 1
    if documented:
        score = 2
    
    if documented and coverage >= 50 and drills >= 1:
        score = 3
    
    if documented and coverage >= 80 and drills >= 2 and trained >= 80:
        score = 4
    
    if documented and coverage >= 95 and drills >= 4 and trained >= 95 and fields.get("equipment_available", False):
        score = 5
    
    return score

def score_communication_protocols(fields):
    """Score communication protocols"""
    documented = fields.get("protocols_documented", False)
    stakeholders = len(fields.get("stakeholders_covered", []))
    channels = fields.get("channels_established_count", 0)
    
    if not documented:
        return 0
    
    score = 1
    if documented:
        score = 2
    
    if documented and stakeholders >= 3 and channels >= 2:
        score = 3
    
    if documented and stakeholders >= 5 and channels >= 3 and fields.get("crisis_communication_plan", False):
        score = 4
    
    if documented and stakeholders >= 7 and channels >= 5 and fields.get("crisis_communication_plan", False) and fields.get("tested_frequency_per_year", 0) >= 1:
        score = 5
    
    return score

def score_early_warning(fields):
    """Score early warning systems"""
    systems = fields.get("systems_deployed_count", 0)
    hazards = len(fields.get("hazards_monitored", []))
    coverage = fields.get("coverage_pct", 0)
    
    if systems == 0:
        return 0
    
    score = 1
    if systems >= 1:
        score = 2
    
    if systems >= 1 and hazards >= 1 and coverage >= 50:
        score = 3
    
    if systems >= 2 and hazards >= 3 and coverage >= 80 and fields.get("integration_with_operations", False):
        score = 4
    
    if systems >= 3 and hazards >= 5 and coverage >= 95 and fields.get("integration_with_operations", False) and fields.get("tested_frequency_per_year", 0) >= 2:
        score = 5
    
    return score

def score_incident_reviews(fields):
    """Score incident reviews & remediation"""
    reviewed = fields.get("incidents_reviewed_count", 0)
    documented = fields.get("lessons_learned_documented", False)
    remediation = fields.get("remediation_plans_count", 0)
    
    if reviewed == 0:
        return 0
    
    score = 1
    if reviewed >= 1:
        score = 2
    
    if reviewed >= 1 and documented:
        score = 3
    
    if reviewed >= 2 and documented and remediation >= 1:
        score = 4
    
    if reviewed >= 3 and documented and remediation >= 2 and fields.get("board_reported", False):
        score = 5
    
    return score

def score_supplier_assessment(fields):
    """Score supplier physical risk assessment"""
    assessed = fields.get("suppliers_assessed_pct", 0)
    critical = fields.get("critical_suppliers_coverage_pct", 0)
    freq = fields.get("assessment_frequency_per_year", 0)
    
    if assessed == 0:
        return 0
    
    score = 1
    if assessed >= 1:
        score = 2
    
    if assessed >= 50 and freq >= 1:
        score = 3
    
    if assessed >= 80 and critical >= 80 and freq >= 1:
        score = 4
    
    if assessed >= 95 and critical >= 95 and freq >= 2 and fields.get("results_tracked", False):
        score = 5
    
    return score

def score_contractual_requirements(fields):
    """Score contractual resilience requirements"""
    covered = fields.get("suppliers_covered_pct", 0)
    documented = fields.get("requirements_documented", False)
    reviewed = fields.get("contracts_reviewed_pct", 0)
    
    if not documented or covered == 0:
        return 0
    
    score = 1
    if documented and covered >= 1:
        score = 2
    
    if documented and covered >= 50 and reviewed >= 50:
        score = 3
    
    if documented and covered >= 80 and reviewed >= 80 and fields.get("compliance_monitored", False):
        score = 4
    
    if documented and covered >= 95 and reviewed >= 95 and fields.get("compliance_monitored", False) and fields.get("critical_suppliers_pct", 0) >= 95:
        score = 5
    
    return score

def score_monitoring_audits(fields):
    """Score monitoring & audits"""
    audits = fields.get("audits_conducted_per_year", 0)
    suppliers = fields.get("suppliers_audited_pct", 0)
    third_party = fields.get("third_party_audits", False)
    
    if audits == 0:
        return 0
    
    score = 1
    if audits >= 1:
        score = 2
    
    if audits >= 1 and suppliers >= 20:
        score = 3
    
    if audits >= 2 and suppliers >= 50 and third_party:
        score = 4
    
    if audits >= 4 and suppliers >= 80 and third_party and fields.get("continuous_monitoring", False):
        score = 5
    
    return score

def score_alternative_sourcing(fields):
    """Score alternative sourcing & buffers"""
    alternatives = fields.get("alternatives_identified_count", 0)
    buffer = fields.get("buffer_inventory_days", 0)
    diversified = fields.get("critical_suppliers_diversified_pct", 0)
    
    if alternatives == 0:
        return 0
    
    score = 1
    if alternatives >= 1:
        score = 2
    
    if alternatives >= 2 and buffer >= 7:
        score = 3
    
    if alternatives >= 3 and buffer >= 14 and diversified >= 50:
        score = 4
    
    if alternatives >= 5 and buffer >= 30 and diversified >= 80 and fields.get("switching_capability_tested", False):
        score = 5
    
    return score

def score_supplier_disclosure(fields):
    """Score supplier disclosure requirements"""
    required = fields.get("suppliers_required_pct", 0)
    frameworks = len(fields.get("disclosure_frameworks", []))
    verified = fields.get("compliance_verified", False)
    
    if required == 0:
        return 0
    
    score = 1
    if required >= 1:
        score = 2
    
    if required >= 50 and frameworks >= 1:
        score = 3
    
    if required >= 80 and frameworks >= 1 and verified:
        score = 4
    
    if required >= 95 and frameworks >= 2 and verified and fields.get("third_party_validation", False):
        score = 5
    
    return score

def score_business_interruption_coverage(fields):
    """Score business interruption coverage"""
    has_bi = fields.get("has_business_interruption", False)
    amount = fields.get("coverage_amount", 0)
    extent = fields.get("coverage_extent_pct", 0)
    
    if not has_bi:
        return 0
    
    score = 1
    if has_bi:
        score = 2
    
    if has_bi and extent >= 50:
        score = 3
    
    if has_bi and extent >= 80 and amount > 0:
        score = 4
    
    if has_bi and extent >= 95 and amount > 0 and fields.get("indemnity_period_days", 0) >= 365:
        score = 5
    
    return score

def score_captives_parametric(fields):
    """Score captives/parametric solutions"""
    captive = fields.get("has_captive", False)
    parametric = fields.get("has_parametric", False)
    amount = fields.get("coverage_amount", 0)
    triggers = len(fields.get("triggers_defined", []))
    
    if not captive and not parametric:
        return 0
    
    score = 1
    if captive or parametric:
        score = 2
    
    if (captive or parametric) and triggers >= 1:
        score = 3
    
    if (captive or parametric) and triggers >= 2 and amount > 0:
        score = 4
    
    if (captive or parametric) and triggers >= 3 and amount > 0 and fields.get("payout_structure", ""):
        score = 5
    
    return score

def score_insurance_gap(fields):
    """Score insurance gap"""
    calculated = fields.get("gap_calculated", False)
    gap = fields.get("total_gap_amount", 0)
    mitigation = fields.get("mitigation_plans_count", 0)
    
    if not calculated:
        return 0
    
    score = 1
    if calculated:
        score = 2
    
    if calculated and gap > 0:
        score = 3
    
    if calculated and gap > 0 and mitigation >= 1:
        score = 4
    
    if calculated and gap > 0 and mitigation >= 2 and fields.get("board_reviewed", False):
        score = 5
    
    return score

def score_internal_controls(fields):
    """Score internal controls over climate data"""
    documented = fields.get("controls_documented", False)
    tested = fields.get("controls_tested_frequency", 0)
    framework = fields.get("data_governance_framework", False)
    
    if not documented:
        return 0
    
    score = 1
    if documented:
        score = 2
    
    if documented and tested >= 1:
        score = 3
    
    if documented and tested >= 2 and framework:
        score = 4
    
    if documented and tested >= 4 and framework and fields.get("automated_controls_count", 0) >= 3:
        score = 5
    
    return score

def score_external_assurance(fields):
    """Score external assurance"""
    provided = fields.get("assurance_provided", False)
    level = fields.get("assurance_level", "")
    scope = fields.get("scope_coverage_pct", 0)
    
    if not provided:
        return 0
    
    score = 1
    if provided:
        score = 2
    
    if provided and scope >= 50:
        score = 3
    
    if provided and scope >= 80 and level and level.lower() in ["reasonable", "high", "limited"]:
        score = 4
    
    if provided and scope >= 95 and level and level.lower() == "reasonable" and fields.get("publicly_disclosed", False):
        score = 5
    
    return score

def score_workforce_heat(fields):
    """Score workforce heat/AQI thresholds & protocols"""
    thresholds = fields.get("thresholds_defined", False)
    protocols = fields.get("protocols_documented", False)
    sites = fields.get("sites_covered_pct", 0)
    workforce = fields.get("workforce_covered_pct", 0)
    
    if not thresholds or not protocols:
        return 0
    
    score = 1
    if thresholds and protocols:
        score = 2
    
    if thresholds and protocols and sites >= 50:
        score = 3
    
    if thresholds and protocols and sites >= 80 and workforce >= 80:
        score = 4
    
    if thresholds and protocols and sites >= 95 and workforce >= 95 and fields.get("compliance_verified", False):
        score = 5
    
    return score

def score_water_security(fields):
    """Score water security/drought controls"""
    controls = fields.get("controls_implemented_count", 0)
    sources = fields.get("water_sources_diversified_count", 0)
    coverage = fields.get("sites_covered_pct", 0)
    
    if controls == 0:
        return 0
    
    score = 1
    if controls >= 1:
        score = 2
    
    if controls >= 2 and coverage >= 50:
        score = 3
    
    if controls >= 3 and sources >= 2 and coverage >= 80:
        score = 4
    
    if controls >= 5 and sources >= 3 and coverage >= 95 and fields.get("monitoring_in_place", False):
        score = 5
    
    return score

def score_community_engagement(fields):
    """Score community/utility engagement"""
    programs = fields.get("engagement_programs_count", 0)
    partnerships = fields.get("partnerships_count", 0)
    stakeholders = len(fields.get("stakeholders_engaged", []))
    
    if programs == 0:
        return 0
    
    score = 1
    if programs >= 1:
        score = 2
    
    if programs >= 1 and stakeholders >= 3:
        score = 3
    
    if programs >= 2 and partnerships >= 2 and stakeholders >= 5:
        score = 4
    
    if programs >= 3 and partnerships >= 3 and stakeholders >= 7 and fields.get("resilience_projects_count", 0) >= 2:
        score = 5
    
    return score

def score_downtime(fields):
    """Score downtime and service impact"""
    tracked = fields.get("downtime_tracked", False)
    target = fields.get("target_downtime_hours", 999)
    actual = fields.get("actual_downtime_hours", 999)
    measured = fields.get("service_impact_measured", False)
    
    if not tracked:
        return 0
    
    score = 1
    if tracked:
        score = 2
    
    if tracked and measured:
        score = 3
    
    if tracked and measured and actual <= target:
        score = 4
    
    if tracked and measured and actual <= target * 0.8 and fields.get("recovery_time_hours", 999) <= 24:
        score = 5
    
    return score

def score_losses(fields):
    """Score losses and loss ratios"""
    tracked = fields.get("losses_tracked", False)
    total = fields.get("total_losses_amount", 0)
    ratio = fields.get("loss_ratio_pct", 0)
    
    if not tracked:
        return 0
    
    score = 1
    if tracked:
        score = 2
    
    if tracked and total >= 0:
        score = 3
    
    if tracked and total >= 0 and ratio >= 0 and fields.get("by_hazard", {}):
        score = 4
    
    if tracked and total >= 0 and ratio >= 0 and fields.get("by_hazard", {}) and fields.get("trend_analysis_done", False):
        score = 5
    
    return score

def score_supplier_disruption(fields):
    """Score supplier disruption days"""
    tracked = fields.get("disruptions_tracked", False)
    total = fields.get("total_disruption_days", 999)
    critical = fields.get("critical_supplier_disruptions", 999)
    
    if not tracked:
        return 0
    
    score = 1
    if tracked:
        score = 2
    
    if tracked and total < 999:
        score = 3
    
    if tracked and total < 30 and critical < 10:
        score = 4
    
    if tracked and total < 7 and critical < 3 and fields.get("mitigation_effectiveness", 0) >= 80:
        score = 5
    
    return score

def score_adaptation_spend(fields):
    """Score adaptation spend and outcomes"""
    tracked = fields.get("spend_tracked", False)
    total = fields.get("total_spend_amount", 0)
    outcomes = fields.get("outcomes_measured", False)
    
    if not tracked:
        return 0
    
    score = 1
    if tracked:
        score = 2
    
    if tracked and total > 0:
        score = 3
    
    if tracked and total > 0 and outcomes:
        score = 4
    
    if tracked and total > 0 and outcomes and fields.get("roi_calculated", False):
        score = 5
    
    return score

def score_measure_by_rubric(measure_name, fields, rubric_text):
    """Generic scoring function using measure-specific functions"""
    # Map measure names to scoring functions
    scoring_functions = {
        "Board-level physical risk oversight": score_board_oversight,
        "Senior management responsibility": score_senior_management,
        "ERM integration of physical risk": score_erm_integration,
        "Physical risk scenario analysis": score_scenario_analysis,
        "Geographic risk mapping": score_geographic_mapping,
        "Business continuity planning coverage": score_bcp_coverage,
        "Insurance program structure": score_insurance_structure,
        "Financial quantification disclosure": score_financial_disclosure,
        "Public adaptation/resilience commitments": score_adaptation_commitments,
        "Incentives linked to physical risk KPIs": score_kpi_incentives,
        "Climate competency and training": score_climate_competency,
        "Disclosure alignment": score_disclosure_alignment,
        "Sector/portfolio risk assessment": score_sector_assessment,
        "Stress testing integration": score_stress_testing,
        "Financial quantification methodology": score_quant_methodology,
        "Dependency mapping": score_dependency_mapping,
        "Tail-risk treatment": score_tail_risk,
        "Data quality & validation": score_data_quality,
        "Facility design standards": score_facility_design,
        "Infrastructure adaptation measures": score_infrastructure_adaptation,
        "Technology system resilience": score_technology_resilience,
        "Location risk policy": score_location_policy,
        "Redundancy and autonomy": score_redundancy,
        "Emergency response procedures": score_emergency_response,
        "Communication protocols": score_communication_protocols,
        "Early warning systems": score_early_warning,
        "Incident reviews & remediation": score_incident_reviews,
        "Supplier physical risk assessment": score_supplier_assessment,
        "Contractual resilience requirements": score_contractual_requirements,
        "Monitoring & audits": score_monitoring_audits,
        "Alternative sourcing & buffers": score_alternative_sourcing,
        "Supplier disclosure requirements": score_supplier_disclosure,
        "Business interruption coverage": score_business_interruption_coverage,
        "Captives/parametric solutions": score_captives_parametric,
        "Insurance gap": score_insurance_gap,
        "Internal controls over climate data": score_internal_controls,
        "External assurance": score_external_assurance,
        "Workforce heat/AQI thresholds & protocols": score_workforce_heat,
        "Water security/drought controls": score_water_security,
        "Community/utility engagement": score_community_engagement,
        "Downtime and service impact": score_downtime,
        "Losses and loss ratios": score_losses,
        "Supplier disruption days": score_supplier_disruption,
        "Adaptation spend and outcomes": score_adaptation_spend,
    }
    
    if measure_name in scoring_functions:
        return scoring_functions[measure_name](fields)
    else:
        # Generic fallback scoring (should not be reached if all measures are implemented)
        # Level 0: no data
        if not fields or all(not v or (isinstance(v, (int, float)) and v == 0) or (isinstance(v, bool) and not v) or (isinstance(v, (list, dict)) and len(v) == 0) for v in fields.values() if v != "evidence"):
            return 0
        # Level 1: some mention
        return 1

def process_measure_for_company(company_id, measure_row, chunks):
    """Process one measure for one company"""
    measure_name = measure_row['Measure']
    definition = measure_row['Definition (physical risk scope)']
    keywords_str = measure_row['Parsing Cues/Keywords']
    rubric = measure_row['Scoring Rubric (0? criteria)']
    
    # Extract keywords
    keywords = extract_keywords(keywords_str)
    if not keywords:
        return {
            "company_id": company_id,
            "measure": measure_name,
            "score": 0,
            "fields": {},
            "error": "No keywords found"
        }
    
    # Find relevant chunks
    relevant_chunks = find_relevant_chunks(chunks, keywords)
    if not relevant_chunks:
        return {
            "company_id": company_id,
            "measure": measure_name,
            "score": 0,
            "fields": {},
            "error": "No relevant chunks found"
        }
    
    # Extract structured data using LLM
    extracted_fields = extract_with_llm(measure_name, definition, relevant_chunks, keywords)
    
    # Score using rubric
    score = score_measure_by_rubric(measure_name, extracted_fields, rubric)
    
    return {
        "company_id": company_id,
        "measure": measure_name,
        "category": measure_row['Category'],
        "score": score,
        "score_percentage": score * 20,  # Convert to 0-100%
        "fields": extracted_fields,
        "keywords_used": keywords[:5],  # Store first 5 keywords
        "chunks_analyzed": len(relevant_chunks)
    }

def run_full_analysis(framework_path, chunks_folder="preprocessed_chunks", output_file="physical_risk_analysis_report.json"):
    """Run full analysis for all companies and measures"""
    print("Loading framework...")
    framework_df = load_framework(framework_path)
    
    all_results = []
    
    # Get list of companies from chunks folder
    company_ids = []
    for file in os.listdir(chunks_folder):
        if file.endswith("_chunks.json"):
            company_id = file.replace("_chunks.json", "")
            company_ids.append(company_id)
    
    print(f"Found {len(company_ids)} companies")
    print(f"Processing {len(framework_df)} measures per company\n")
    
    for company_id in company_ids:
        print(f"\n{'='*80}")
        print(f"Processing company: {company_id}")
        print(f"{'='*80}")
        
        # Load chunks for this company
        chunks = load_company_chunks(company_id, chunks_folder)
        if not chunks:
            print(f"  No chunks found for {company_id}, skipping...")
            continue
        
        print(f"  Loaded {len(chunks)} chunks")
        
        # Process each measure
        for idx, measure_row in tqdm(framework_df.iterrows(), total=len(framework_df), desc=f"  Processing measures"):
            try:
                result = process_measure_for_company(company_id, measure_row, chunks)
                all_results.append(result)
                
                if result["score"] > 0:
                    print(f"    [{measure_row['Category']}] {measure_row['Measure']}: Score {result['score']}/5")
            except Exception as e:
                print(f"    Error processing {measure_row['Measure']}: {e}")
                all_results.append({
                    "company_id": company_id,
                    "measure": measure_row['Measure'],
                    "category": measure_row['Category'],
                    "score": 0,
                    "error": str(e)
                })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Also create Excel report
    results_df = pd.DataFrame(all_results)
    excel_file = output_file.replace('.json', '.xlsx')
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Summary by category
        if 'category' in results_df.columns and 'score' in results_df.columns:
            summary = results_df.groupby(['company_id', 'category'])['score'].agg(['mean', 'max', 'min', 'count']).reset_index()
            summary.columns = ['Company_ID', 'Category', 'Avg_Score', 'Max_Score', 'Min_Score', 'Measure_Count']
            summary.to_excel(writer, sheet_name='Summary_by_Category', index=False)
        
        # Overall summary
        overall = results_df.groupby('company_id')['score'].agg(['mean', 'max', 'min']).reset_index()
        overall.columns = ['Company_ID', 'Avg_Score', 'Max_Score', 'Min_Score']
        overall.to_excel(writer, sheet_name='Overall_Summary', index=False)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"Excel report saved to: {excel_file}")
    print(f"Total results: {len(all_results)}")
    print(f"Average score: {results_df['score'].mean():.2f}/5")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Check if preprocessing was done
    if not os.path.exists("preprocessed_chunks"):
        print("Error: Preprocessed chunks not found!")
        print("Please run phase1_preprocess.py first")
        sys.exit(1)
    
    run_full_analysis(
        framework_path='PhysicalRisk_Resilience_Framework.xlsx',
        chunks_folder='preprocessed_chunks',
        output_file='physical_risk_analysis_report.json'
    )
