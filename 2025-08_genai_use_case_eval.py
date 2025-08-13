#!/usr/bin/env python3
"""
AI Use Case Commoditization Scorer - Standalone Version
Evaluates JIRA AI use cases for commoditization risk using Bedrock Claude
"""

import json
import time
import logging
import boto3
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# ============================================================================
# Configuration
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Bedrock settings
BEDROCK_REGION = os.environ.get('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Rate limiting (seconds between calls)
BEDROCK_DELAY = 1.0
WEB_SEARCH_DELAY = 0.5

# Web proxy (optional)
HTTP_PROXY = os.environ.get('HTTP_PROXY', None)
HTTPS_PROXY = os.environ.get('HTTPS_PROXY', None)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class UseCase:
    """Represents an AI use case from JIRA"""
    id: str
    title: str
    description: str
    business_unit: str = "Unknown"
    technical_details: str = ""
    data_sources: str = ""
    status: str = "Unknown"


@dataclass
class EvaluationResult:
    """Complete evaluation result for a use case"""
    use_case_id: str
    title: str

    # Individual scores (1-5)
    time_to_value: int
    duration_of_value: int
    technical_feasibility: int
    generic_capability_risk: int
    vendor_overlap_risk: int

    # Calculated fields
    total_score: int
    commoditization_risk_percent: float

    # Analysis
    vendor_threats: List[str]
    github_signals: Dict
    blind_spots: List[str]
    recommendation: str
    reasoning: str

    # Metadata
    evaluation_timestamp: str
    confidence_score: float


# ============================================================================
# Bedrock Claude Integration
# ============================================================================

class BedrockEvaluator:
    """Handles evaluation using Claude on Amazon Bedrock"""

    def __init__(self):
        """Initialize Bedrock client"""
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=BEDROCK_REGION
            )
            logger.info(f"Initialized Bedrock client in {BEDROCK_REGION}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def evaluate_use_case(self, use_case: UseCase, market_context: Dict) -> Dict:
        """
        Evaluate a single use case using Claude

        Args:
            use_case: The use case to evaluate
            market_context: Additional context from web searches

        Returns:
            Dict with scoring results
        """

        prompt = self._create_evaluation_prompt(use_case, market_context)

        try:
            # Call Bedrock API
            response = self.client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.2,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                })
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except Exception as e:
            logger.error(f"Bedrock evaluation failed for {use_case.id}: {e}")
            return self._get_default_scores()

    def _create_evaluation_prompt(self, use_case: UseCase, market_context: Dict) -> str:
        """Create the evaluation prompt for Claude"""

        return f"""You are an enterprise AI strategist evaluating AI use cases for commoditization risk.

CONTEXT:
- Current date: {datetime.now().strftime('%Y-%m-%d')}
- Major AI platforms: Microsoft Copilot (70% of Fortune 500), Salesforce Agentforce, ServiceNow Now Assist
- Open source momentum: LangChain (88k stars), AutoGPT (160k stars)
- Commoditization timeline: Basic ML (already done), NLP/Vision (12-18 months), Complex workflows (2-3 years)

USE CASE TO EVALUATE:
ID: {use_case.id}
Title: {use_case.title}
Description: {use_case.description}
Business Unit: {use_case.business_unit}
Technical Details: {use_case.technical_details or "Not specified"}
Data Sources: {use_case.data_sources or "Not specified"}

MARKET INTELLIGENCE:
{json.dumps(market_context, indent=2)}

EVALUATION TASK:
Score this use case on 5 dimensions (1=worst, 5=best):

1. Time to Value: How quickly can this deliver value?
   - 5: <3 months using existing tools
   - 3: 6-12 months with integration work  
   - 1: >12 months, needs R&D

2. Duration of Value: How long will this remain valuable?
   - 5: Multi-year strategic advantage
   - 3: 1-2 years before commoditization
   - 1: <6 months before obsolete

3. Technical Feasibility: How ready is the technology?
   - 5: Off-the-shelf APIs available
   - 3: Needs some custom development
   - 1: Requires breakthrough research

4. Generic Capability Risk: Will this become a generic feature?
   - 5: Highly specific to our business
   - 3: Somewhat unique, may commoditize
   - 1: Already exists in Copilot/Claude/ChatGPT

5. Vendor Overlap Risk: Are vendors building this?
   - 5: No vendor will ever build this
   - 3: On vendor roadmaps for 2026+
   - 1: Already in vendor products

Also identify:
- Which vendors pose the biggest threat (Microsoft, Salesforce, ServiceNow, Google, AWS, etc.)
- GitHub projects that could replace this (with star counts if known)
- Blind spots the evaluation team might miss
- Your confidence in this assessment (0.0-1.0)

Respond ONLY with valid JSON in this exact format:
{{
  "time_to_value": <int 1-5>,
  "duration_of_value": <int 1-5>,
  "technical_feasibility": <int 1-5>,
  "generic_capability_risk": <int 1-5>,
  "vendor_overlap_risk": <int 1-5>,
  "vendor_threats": ["vendor1", "vendor2"],
  "competing_github_projects": ["project1 (Xk stars)", "project2"],
  "blind_spots": ["blind spot 1", "blind spot 2"],
  "confidence": <float 0-1>,
  "reasoning": "Brief explanation of the scoring"
}}"""

    def _get_default_scores(self) -> Dict:
        """Return default scores if evaluation fails"""
        return {
            "time_to_value": 3,
            "duration_of_value": 3,
            "technical_feasibility": 3,
            "generic_capability_risk": 3,
            "vendor_overlap_risk": 3,
            "vendor_threats": ["Unknown - manual review needed"],
            "competing_github_projects": [],
            "blind_spots": ["Automated evaluation failed - requires manual review"],
            "confidence": 0.1,
            "reasoning": "Default scores due to evaluation error"
        }


# ============================================================================
# Web Search for Market Signals
# ============================================================================

class MarketSignalCollector:
    """Collects market signals from web searches"""

    def __init__(self, proxy_url: Optional[str] = None):
        """Initialize with optional proxy"""
        self.session = requests.Session()
        if proxy_url:
            self.session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            logger.info(f"Using proxy: {proxy_url}")

    def gather_signals(self, use_case: UseCase) -> Dict:
        """
        Gather market signals for a use case

        Returns:
            Dict containing GitHub signals, vendor info, etc.
        """

        signals = {
            "search_terms": [],
            "github_activity": {},
            "vendor_mentions": [],
            "technology_maturity": "unknown"
        }

        # Extract key terms from the use case
        search_terms = self._extract_search_terms(use_case)
        signals["search_terms"] = search_terms

        # Search for GitHub projects (simplified - in production use GitHub API)
        for term in search_terms[:3]:  # Limit to avoid rate limits
            try:
                # This is where you'd make actual web searches
                # For now, we'll use heuristics based on common patterns
                signals["github_activity"][term] = self._estimate_github_activity(term)
                time.sleep(WEB_SEARCH_DELAY)
            except Exception as e:
                logger.warning(f"Search failed for '{term}': {e}")

        # Check vendor overlap based on keywords
        signals["vendor_mentions"] = self._check_vendor_keywords(use_case)

        # Estimate technology maturity
        signals["technology_maturity"] = self._estimate_maturity(use_case)

        return signals

    def _extract_search_terms(self, use_case: UseCase) -> List[str]:
        """Extract search terms from use case"""

        # Key AI/ML terms to look for
        ai_keywords = [
            'nlp', 'natural language', 'llm', 'gpt', 'claude', 'gemini',
            'computer vision', 'ocr', 'image recognition',
            'prediction', 'forecast', 'classification',
            'automation', 'rpa', 'workflow',
            'chatbot', 'virtual agent', 'conversational',
            'rag', 'retrieval', 'embedding', 'vector',
            'sentiment', 'entity extraction', 'summarization'
        ]

        text = f"{use_case.title} {use_case.description}".lower()
        found_terms = []

        for keyword in ai_keywords:
            if keyword in text:
                found_terms.append(keyword)

        # Add specific words from title
        title_words = use_case.title.lower().split()
        for word in title_words:
            if len(word) > 4 and word not in ['system', 'using', 'based', 'create', 'build']:
                found_terms.append(word)

        return list(set(found_terms))[:5]  # Return unique terms, max 5

    def _estimate_github_activity(self, term: str) -> Dict:
        """Estimate GitHub activity for a search term"""

        # Common high-activity projects (simplified heuristics)
        high_activity = {
            'llm': {'projects': 50, 'max_stars': 88000, 'growth': 'high'},
            'chatbot': {'projects': 100, 'max_stars': 45000, 'growth': 'high'},
            'rag': {'projects': 30, 'max_stars': 25000, 'growth': 'very high'},
            'nlp': {'projects': 200, 'max_stars': 65000, 'growth': 'medium'},
            'computer vision': {'projects': 150, 'max_stars': 55000, 'growth': 'medium'},
            'embedding': {'projects': 40, 'max_stars': 30000, 'growth': 'high'},
            'automation': {'projects': 300, 'max_stars': 160000, 'growth': 'high'}
        }

        return high_activity.get(term, {'projects': 10, 'max_stars': 1000, 'growth': 'low'})

    def _check_vendor_keywords(self, use_case: UseCase) -> List[str]:
        """Check for vendor-specific keywords"""

        vendor_keywords = {
            'Microsoft Copilot': ['email', 'document', 'excel', 'powerpoint', 'teams', 'outlook'],
            'Salesforce Einstein': ['crm', 'sales', 'customer', 'lead', 'opportunity', 'forecast'],
            'ServiceNow': ['ticket', 'incident', 'service desk', 'it support', 'workflow'],
            'Google Vertex': ['prediction', 'classification', 'vision', 'translation'],
            'AWS Bedrock': ['claude', 'llm', 'foundation model', 'generation']
        }

        text = f"{use_case.title} {use_case.description}".lower()
        threats = []

        for vendor, keywords in vendor_keywords.items():
            if any(kw in text for kw in keywords):
                threats.append(vendor)

        return threats

    def _estimate_maturity(self, use_case: UseCase) -> str:
        """Estimate technology maturity level"""

        text = f"{use_case.title} {use_case.description}".lower()

        # Commoditized capabilities
        if any(term in text for term in ['email draft', 'document summary', 'translation',
                                         'sentiment analysis', 'chat support']):
            return "commoditized"

        # Maturing capabilities
        if any(term in text for term in ['rag', 'knowledge base', 'classification',
                                         'entity extraction', 'forecast']):
            return "maturing"

        # Emerging capabilities
        if any(term in text for term in ['agent', 'autonomous', 'reasoning',
                                         'multi-modal', 'complex workflow']):
            return "emerging"

        return "unknown"


# ============================================================================
# Main Evaluation System
# ============================================================================

class CommoditizationScorer:
    """Main system for scoring AI use cases"""

    def __init__(self, use_proxy: bool = False):
        """Initialize the scoring system"""
        self.bedrock = BedrockEvaluator()

        proxy = HTTP_PROXY if use_proxy and HTTP_PROXY else None
        self.market_collector = MarketSignalCollector(proxy)

        self.results = []

    def process_jira_file(self, filepath: str, output_file: str = None) -> List[EvaluationResult]:
        """
        Process a JIRA JSON export file

        Args:
            filepath: Path to JIRA JSON file
            output_file: Optional output file path

        Returns:
            List of evaluation results
        """

        logger.info(f"Loading JIRA data from {filepath}")
        use_cases = self._load_jira_data(filepath)
        logger.info(f"Loaded {len(use_cases)} use cases")

        # Process in parallel with thread pool
        max_workers = 3  # Limit concurrent Bedrock calls

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluations
            future_to_usecase = {
                executor.submit(self._evaluate_single, uc): uc
                for uc in use_cases
            }

            # Collect results as they complete
            for future in as_completed(future_to_usecase):
                use_case = future_to_usecase[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    logger.info(f"Evaluated {use_case.id}: {result.recommendation}")
                except Exception as e:
                    logger.error(f"Failed to evaluate {use_case.id}: {e}")

        # Sort results by commoditization risk
        self.results.sort(key=lambda x: x.commoditization_risk_percent, reverse=True)

        # Save results if output file specified
        if output_file:
            self._save_results(output_file)

        # Print summary
        self._print_summary()

        return self.results

    def _load_jira_data(self, filepath: str) -> List[UseCase]:
        """Load and parse JIRA JSON export"""

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        use_cases = []

        # Handle different JIRA export formats
        if isinstance(data, dict):
            issues = data.get('issues', [])
        elif isinstance(data, list):
            issues = data
        else:
            raise ValueError(f"Unexpected JIRA format: {type(data)}")

        for issue in issues:
            # Extract fields (adjust based on your JIRA configuration)
            if isinstance(issue, dict):
                fields = issue.get('fields', issue)

                use_case = UseCase(
                    id=issue.get('key', issue.get('id', f"UC-{len(use_cases)}")),
                    title=fields.get('summary', 'Untitled'),
                    description=fields.get('description', ''),
                    business_unit=fields.get('customfield_10001', 'Unknown'),
                    technical_details=fields.get('customfield_10002', ''),
                    data_sources=fields.get('customfield_10003', ''),
                    status=fields.get('status', {}).get('name', 'Unknown') if isinstance(fields.get('status'),
                                                                                         dict) else 'Unknown'
                )

                use_cases.append(use_case)

        return use_cases

    def _evaluate_single(self, use_case: UseCase) -> EvaluationResult:
        """Evaluate a single use case"""

        logger.info(f"Evaluating: {use_case.title}")

        # Gather market signals
        market_signals = self.market_collector.gather_signals(use_case)

        # Get Bedrock evaluation
        eval_result = self.bedrock.evaluate_use_case(use_case, market_signals)

        # Calculate total score
        total_score = sum([
            eval_result['time_to_value'],
            eval_result['duration_of_value'],
            eval_result['technical_feasibility'],
            eval_result['generic_capability_risk'],
            eval_result['vendor_overlap_risk']
        ])

        # Calculate commoditization risk percentage
        # Lower scores = higher risk
        risk_factors = [
            (6 - eval_result['generic_capability_risk']) / 5 * 0.35,  # 35% weight
            (6 - eval_result['vendor_overlap_risk']) / 5 * 0.35,  # 35% weight
            (6 - eval_result['duration_of_value']) / 5 * 0.20,  # 20% weight
            (6 - eval_result['technical_feasibility']) / 5 * 0.10  # 10% weight
        ]
        commoditization_risk = sum(risk_factors) * 100

        # Generate recommendation
        recommendation = self._generate_recommendation(total_score, commoditization_risk)

        # Add rate limiting
        time.sleep(BEDROCK_DELAY)

        return EvaluationResult(
            use_case_id=use_case.id,
            title=use_case.title,
            time_to_value=eval_result['time_to_value'],
            duration_of_value=eval_result['duration_of_value'],
            technical_feasibility=eval_result['technical_feasibility'],
            generic_capability_risk=eval_result['generic_capability_risk'],
            vendor_overlap_risk=eval_result['vendor_overlap_risk'],
            total_score=total_score,
            commoditization_risk_percent=round(commoditization_risk, 1),
            vendor_threats=eval_result.get('vendor_threats', []),
            github_signals=market_signals,
            blind_spots=eval_result.get('blind_spots', []),
            recommendation=recommendation,
            reasoning=eval_result.get('reasoning', ''),
            evaluation_timestamp=datetime.now().isoformat(),
            confidence_score=eval_result.get('confidence', 0.5)
        )

    def _generate_recommendation(self, total_score: int, risk_percent: float) -> str:
        """Generate recommendation based on scores"""

        if risk_percent > 70:
            return "🛑 STOP - High commoditization risk. Use vendor solution or open source."
        elif risk_percent > 50:
            if total_score >= 18:
                return "⚡ ACCELERATE - Good opportunity but move fast before commoditization"
            else:
                return "⚠️ RECONSIDER - Moderate value with high commoditization risk"
        elif total_score >= 20:
            return "✅ INVEST - Strong differentiation opportunity. Proceed with confidence."
        elif total_score >= 15:
            return "🎯 SELECTIVE - Proceed if strategic alignment is strong"
        else:
            return "❌ AVOID - Low value proposition"

    def _save_results(self, output_file: str):
        """Save results to JSON file"""

        output = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_use_cases': len(self.results),
                'model': BEDROCK_MODEL_ID,
                'high_risk_count': len([r for r in self.results if r.commoditization_risk_percent > 70]),
                'recommended_investments': len([r for r in self.results if 'INVEST' in r.recommendation])
            },
            'results': [asdict(r) for r in self.results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to {output_file}")

        # Also save CSV for Excel
        csv_file = output_file.replace('.json', '.csv')
        self._save_csv(csv_file)

    def _save_csv(self, csv_file: str):
        """Save results as CSV for Excel analysis"""

        import csv

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if not self.results:
                return

            fieldnames = [
                'use_case_id', 'title', 'total_score', 'commoditization_risk_percent',
                'recommendation', 'vendor_threats', 'confidence_score'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                writer.writerow({
                    'use_case_id': result.use_case_id,
                    'title': result.title,
                    'total_score': result.total_score,
                    'commoditization_risk_percent': result.commoditization_risk_percent,
                    'recommendation': result.recommendation,
                    'vendor_threats': ', '.join(result.vendor_threats),
                    'confidence_score': result.confidence_score
                })

        logger.info(f"CSV saved to {csv_file}")

    def _print_summary(self):
        """Print evaluation summary"""

        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 70)
        print("COMMODITIZATION RISK ASSESSMENT SUMMARY")
        print("=" * 70)

        # Risk distribution
        high_risk = [r for r in self.results if r.commoditization_risk_percent > 70]
        medium_risk = [r for r in self.results if 40 <= r.commoditization_risk_percent <= 70]
        low_risk = [r for r in self.results if r.commoditization_risk_percent < 40]

        print(f"\n📊 Risk Distribution ({len(self.results)} use cases):")
        print(f"  🔴 High Risk (>70%):    {len(high_risk):3d} use cases")
        print(f"  🟡 Medium Risk (40-70%): {len(medium_risk):3d} use cases")
        print(f"  🟢 Low Risk (<40%):     {len(low_risk):3d} use cases")

        # Top recommendations
        invest = [r for r in self.results if 'INVEST' in r.recommendation]
        stop = [r for r in self.results if 'STOP' in r.recommendation]

        print(f"\n💡 Recommendations:")
        print(f"  ✅ Recommended investments: {len(invest)}")
        print(f"  🛑 Should stop/redirect:    {len(stop)}")

        # Highest risk cases
        print(f"\n⚠️  Highest Commoditization Risks:")
        for r in self.results[:5]:
            if r.commoditization_risk_percent > 50:
                vendors = ', '.join(r.vendor_threats[:2]) if r.vendor_threats else 'Multiple'
                print(f"  - {r.title[:40]:40s} | Risk: {r.commoditization_risk_percent:4.1f}% | Threats: {vendors}")

        # Best opportunities
        print(f"\n🎯 Best Investment Opportunities:")
        best = sorted([r for r in self.results if r.commoditization_risk_percent < 50],
                      key=lambda x: x.total_score, reverse=True)
        for r in best[:5]:
            print(f"  - {r.title[:40]:40s} | Score: {r.total_score}/25 | Risk: {r.commoditization_risk_percent:4.1f}%")

        # Vendor threats
        all_vendors = []
        for r in self.results:
            all_vendors.extend(r.vendor_threats)

        if all_vendors:
            from collections import Counter
            vendor_counts = Counter(all_vendors)
            print(f"\n🏢 Top Vendor Threats:")
            for vendor, count in vendor_counts.most_common(5):
                print(f"  - {vendor}: threatens {count} use cases")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main execution function"""

    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate AI use cases for commoditization risk using AWS Bedrock'
    )

    parser.add_argument(
        'jira_file',
        help='Path to JIRA JSON export file'
    )

    parser.add_argument(
        '--output',
        '-o',
        default='commoditization_assessment.json',
        help='Output file for results (default: commoditization_assessment.json)'
    )

    parser.add_argument(
        '--use-proxy',
        action='store_true',
        help='Use HTTP/HTTPS proxy from environment variables'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check for AWS credentials
    if not boto3.Session().get_credentials():
        print("ERROR: AWS credentials not found!")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("Or configure AWS CLI with: aws configure")
        sys.exit(1)

    # Run evaluation
    try:
        scorer = CommoditizationScorer(use_proxy=args.use_proxy)
        results = scorer.process_jira_file(args.jira_file, args.output)

        print(f"\n✅ Evaluation complete!")
        print(f"   Results saved to: {args.output}")
        print(f"   CSV saved to: {args.output.replace('.json', '.csv')}")

    except FileNotFoundError:
        print(f"ERROR: File not found: {args.jira_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()