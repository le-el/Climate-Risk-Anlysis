# Measure-Specific Implementations

This document lists all measures with custom JSON schemas and scoring functions.

## Fully Implemented Measures (44 total - 100% coverage! ✅)

**All 44 measures now have custom JSON schemas and scoring functions implemented.**

### 1. Board-level physical risk oversight
**Schema Fields:**
- committee_name, charter_url, charter_last_updated_date
- review_frequency_per_year, hazards_count, horizons
- subsidiaries_coverage_pct

**Scoring Logic:**
- Level 0: No board reference
- Level 1: Mentions climate oversight
- Level 2: Named committee but <50% coverage or ad hoc
- Level 3: Explicit oversight, ≥ annual, ≥50% coverage
- Level 4: ≥2x/year, ≥80% coverage, multi-hazard, all horizons
- Level 5: ≥ quarterly, ≥95% coverage

---

### 2. Senior management responsibility
**Schema Fields:**
- executive_name, executive_title, reporting_line
- team_size_ftes, has_budget, has_kpis
- reporting_frequency_per_year, scope_coverage_pct

**Scoring Logic:**
- Level 0: No named executive
- Level 1: Named executive
- Level 2: Named executive with reporting line
- Level 3: ≥ annual reporting, ≥50% scope
- Level 4: Budget + KPIs, ≥2x/year, ≥80% scope
- Level 5: Budget + KPIs + team, ≥2x/year, ≥95% scope

---

### 3. ERM integration of physical risk
**Schema Fields:**
- in_risk_taxonomy, has_risk_appetite, has_tolerances
- in_risk_register, linked_to_strategy, linked_to_capex
- update_frequency_per_year, functions_covered, multi_hazard_analysis

**Scoring Logic:**
- Level 0: Not in ERM
- Level 1: Mentioned
- Level 2: In taxonomy or has appetite
- Level 3: In taxonomy + tolerances + ≥1 function
- Level 4: Linked to strategy + ≥3 functions + multi-hazard
- Level 5: Linked to strategy + capex + ≥5 functions

---

### 4. Physical risk scenario analysis
**Schema Fields:**
- scenarios_count, scenario_types, horizons_analyzed
- assets_coverage_pct, hazards_analyzed, vendor_models_used
- return_periods_included, frequency_per_year

**Scoring Logic:**
- Level 0: No scenarios
- Level 1: Any mention
- Level 2: ≥1 scenario
- Level 3: ≥2 scenarios, ≥50% assets, ≥1 hazard
- Level 4: ≥3 scenarios, ≥80% assets, ≥2 hazards, ≥2 horizons
- Level 5: ≥3 scenarios, ≥95% assets, ≥3 hazards, ≥3 horizons, return periods

---

### 5. Geographic risk mapping
**Schema Fields:**
- assets_geocoded_pct, hazards_mapped, update_frequency_per_year
- has_gis_system, horizons_mapped, vendor_used, validation_done

**Scoring Logic:**
- Level 0: No mapping
- Level 1: Any mention
- Level 2: <50% geocoded or <1 hazard
- Level 3: ≥50% geocoded, ≥3 hazards, ≥1 update/year
- Level 4: ≥80% geocoded, ≥5 hazards, GIS system, ≥2 horizons
- Level 5: ≥95% geocoded, ≥5 hazards, GIS system, validated

---

### 6. Business continuity planning coverage
**Schema Fields:**
- coverage_pct, facilities_covered, has_bcp_document
- tested_frequency_per_year, includes_physical_risk, multi_hazard_coverage

**Scoring Logic:**
- Level 0: No BCP document
- Level 1: Any mention
- Level 2: Has BCP document
- Level 3: Document + ≥50% coverage + includes physical risk
- Level 4: ≥80% coverage + tested + multi-hazard
- Level 5: ≥95% coverage + tested + multi-hazard

---

### 7. Insurance program structure
**Schema Fields:**
- has_property_insurance, has_business_interruption
- coverage_amount, deductible_amount, insurer_names
- renewal_frequency, includes_climate_risk

**Scoring Logic:**
- Level 0: No property insurance
- Level 1: Any mention
- Level 2: Has property insurance
- Level 3: Property + business interruption
- Level 4: Property + BI + includes climate risk
- Level 5: Property + BI + climate risk + coverage amount disclosed

---

### 8. Financial quantification disclosure
**Schema Fields:**
- exposure_by_peril, exposure_by_region, insured_vs_uninsured
- business_interruption_included, horizons_disclosed
- confidence_intervals, aal_disclosed, pml_disclosed

**Scoring Logic:**
- Level 0: No exposure data
- Level 1: Any mention
- Level 2: Exposure by peril or region
- Level 3: Exposure data + ≥1 horizon
- Level 4: By peril + region + insured/uninsured + ≥2 horizons
- Level 5: Full disclosure + BI + ≥3 horizons + AAL/PML

---

### 9. Public adaptation/resilience commitments
**Schema Fields:**
- has_targets, target_description, budget_amount
- capex_allocated, time_bound, target_date
- sites_coverage_pct, multi_hazard

**Scoring Logic:**
- Level 0: No targets
- Level 1: Any mention
- Level 2: Has targets
- Level 3: Time-bound targets + budget/capex + ≥50% sites
- Level 4: Time-bound + budget/capex + ≥80% sites + multi-hazard
- Level 5: Time-bound + budget/capex + ≥95% sites + multi-hazard

---

### 10. Incentives linked to physical risk KPIs
**Schema Fields:**
- has_kpi_linkage, kpi_description, compensation_weight_pct
- executives_covered_pct, kpis_are_measurable, has_audit

**Scoring Logic:**
- Level 0: No KPI linkage
- Level 1: Any mention
- Level 2: Has linkage but not measurable
- Level 3: Measurable KPIs + ≥10% weight + ≥50% executives
- Level 4: Measurable + ≥10% weight + ≥80% executives
- Level 5: Measurable + ≥10% weight + ≥95% executives + audited

---

## All Additional Measures Implemented (34 additional measures)

The following 34 measures now have full implementations with custom schemas and scoring:

11. **Climate competency and training** - Training programs, employee coverage, frequency
12. **Disclosure alignment** - Framework alignment, metrics, assurance
13. **Sector/portfolio risk assessment** - Sectors assessed, frequency, hazards
14. **Stress testing integration** - Stress tests, scenarios, physical risk inclusion
15. **Financial quantification methodology** - Methodology documentation, validation, models
16. **Dependency mapping** - Dependencies mapped, critical deps, updates
17. **Tail-risk treatment** - Tail risks, treatment methods, monitoring
18. **Data quality & validation** - Validation procedures, third-party validation, controls
19. **Facility design standards** - Standards documentation, application coverage, certification
20. **Infrastructure adaptation measures** - Measures implemented, investment, effectiveness
21. **Technology system resilience** - Systems assessed, backup coverage, disaster recovery
22. **Location risk policy** - Policy documentation, assessments, board approval
23. **Redundancy and autonomy** - Redundant systems, autonomous systems, coverage
24. **Emergency response procedures** - Procedures, drills, training coverage
25. **Communication protocols** - Protocols, stakeholders, crisis plans
26. **Early warning systems** - Systems deployed, hazards monitored, integration
27. **Incident reviews & remediation** - Incidents reviewed, lessons learned, remediation
28. **Supplier physical risk assessment** - Suppliers assessed, critical suppliers, frequency
29. **Contractual resilience requirements** - Suppliers covered, compliance monitoring
30. **Monitoring & audits** - Audits conducted, third-party audits, continuous monitoring
31. **Alternative sourcing & buffers** - Alternatives identified, buffer inventory, diversification
32. **Supplier disclosure requirements** - Requirements, frameworks, verification
33. **Business interruption coverage** - BI coverage, amount, indemnity period
34. **Captives/parametric solutions** - Captives, parametric, triggers, payout structure
35. **Insurance gap** - Gap calculation, mitigation plans, board review
36. **Internal controls over climate data** - Controls documented, tested, framework
37. **External assurance** - Assurance level, scope, public disclosure
38. **Workforce heat/AQI thresholds & protocols** - Thresholds, protocols, coverage
39. **Water security/drought controls** - Controls implemented, sources diversified
40. **Community/utility engagement** - Engagement programs, partnerships, projects
41. **Downtime and service impact** - Downtime tracked, targets vs actual
42. **Losses and loss ratios** - Losses tracked, ratios, trend analysis
43. **Supplier disruption days** - Disruptions tracked, critical supplier impacts
44. **Adaptation spend and outcomes** - Spend tracked, outcomes measured, ROI

---

## Adding New Measures

To add a new measure-specific implementation:

1. **Add schema** to `get_json_schema_for_measure()` function
2. **Add scoring function** (e.g., `score_measure_name()`)
3. **Register** in `score_measure_by_rubric()` mapping

The generic fallback will handle all other measures with basic scoring (0 or 1).
