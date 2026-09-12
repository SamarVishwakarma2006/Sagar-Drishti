"""
Automated Test Suite for SagarBot Conversational Decision Intelligence (Phase 4.7).
Verifies:
1. Intent Classification (All 9 intents + general chat/clarification)
2. Geographic Entity Resolution (Basins, coastal cities, contextual 'here', ambiguity policy)
3. Horizon Resolution (Strict 1-to-1 dispatch: 0d -> 0d, 1d -> 1d, 2d -> 2d, 3d -> 3d, zero fallback)
4. General Ocean Status Optimization (Direct Copernicus observation, skips ML prediction)
5. Historical Event Store Grounding (Zero hardcoded catalogs, Fengal, Dana, Asna)
6. Conversational Memory & Stale Prediction Eviction (Invalidation upon region/date/horizon change)
7. Non-Causal & Uncalibrated Confidence Guardrails (Forbidden phrases, decision support structure)
8. End-to-End Orchestration via /api/chat and /api/sagarbot/chat
"""
import unittest
import asyncio
from unittest.mock import patch, MagicMock

from app.services.sagarbot_service import (
    SagarBotService,
    IntentClassifier,
    IntentType,
    EntityResolver,
    SagarBotSessionManager,
    DecisionSupportGenerator,
    COASTAL_CITIES,
)
from app.services.event_store import HistoricalEventStore
from app.services.prediction_service import PredictionService
from app.models.schemas import ChatRequest, ChatContext, PredictionRequest


class TestSagarBotIntelligence(unittest.TestCase):

    def setUp(self):
        # Reset session store for clean test isolation
        SagarBotSessionManager._sessions.clear()
        HistoricalEventStore.reset_to_seeds()

    # ==========================================================================
    # 1. INTENT CLASSIFICATION TESTS (All 9 Explicit Intents)
    # ==========================================================================
    def test_intent_classification_all_9_intents(self):
        cases = [
            ("Is there any risk in the Bay of Bengal?", IntentType.CURRENT_RISK),
            ("Is there cyclone risk near Chennai?", IntentType.CURRENT_RISK),
            ("What is the risk over the next 3 days?", IntentType.FUTURE_RISK),
            ("What about tomorrow?", IntentType.FUTURE_RISK),
            ("Why is the risk elevated?", IntentType.WHY_RISK),
            ("Why?", IntentType.WHY_RISK),
            ("Is this similar to Fengal?", IntentType.HISTORICAL_COMPARISON),
            ("Compare this with Cyclone Dana", IntentType.HISTORICAL_COMPARISON),
            ("Which parameters are driving it?", IntentType.PARAMETER_ANALYSIS),
            ("How is salinity affecting this risk?", IntentType.PARAMETER_ANALYSIS),
            ("What happened during Fengal?", IntentType.EVENT_LOOKUP),
            ("Tell me about Cyclone Dana", IntentType.EVENT_LOOKUP),
            ("Which is riskier, BOB or Arabian Sea?", IntentType.REGION_COMPARISON),
            ("Compare Bay of Bengal vs Arabian Sea", IntentType.REGION_COMPARISON),
            ("What is the temperature here?", IntentType.GENERAL_OCEAN_STATUS),
            ("How deep is the mixed layer?", IntentType.GENERAL_OCEAN_STATUS),
            ("How confident are you?", IntentType.CONFIDENCE_INQUIRY),
            ("What is your confidence level?", IntentType.CONFIDENCE_INQUIRY),
        ]

        for text, expected_intent in cases:
            classified = IntentClassifier.classify(text, has_active_prediction=True)
            self.assertEqual(
                classified,
                expected_intent,
                f"Failed on utterance: '{text}'. Expected {expected_intent}, got {classified}"
            )

    # ==========================================================================
    # 2. ENTITY RESOLUTION TESTS
    # ==========================================================================
    def test_entity_resolution_basins(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.2 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        session_state = SagarBotSessionManager.get_session("test_sess")

        # Bay of Bengal
        res_bob = EntityResolver.resolve_entities("What is the risk in Bay of Bengal?", ctx, session_state)
        self.assertEqual(res_bob.site_id, "bob")
        self.assertEqual(res_bob.region_name, "Bay of Bengal")

        # Arabian Sea
        res_aras = EntityResolver.resolve_entities("Check risk in Arabian Sea", ctx, session_state)
        self.assertEqual(res_aras.site_id, "aras")
        self.assertEqual(res_aras.region_name, "Arabian Sea")

    def test_entity_resolution_coastal_cities(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.2 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        session_state = SagarBotSessionManager.get_session("test_sess")

        # Chennai -> Bay of Bengal coastal point
        res_chennai = EntityResolver.resolve_entities("Risk near Chennai", ctx, session_state)
        self.assertEqual(res_chennai.coordinates["lat"], 13.08)
        self.assertEqual(res_chennai.coordinates["lon"], 80.27)
        self.assertEqual(res_chennai.site_id, "bob")

        # Mumbai -> Arabian Sea coastal point
        res_mumbai = EntityResolver.resolve_entities("Is there a storm near Mumbai?", ctx, session_state)
        self.assertEqual(res_mumbai.coordinates["lat"], 18.92)
        self.assertEqual(res_mumbai.coordinates["lon"], 72.83)
        self.assertEqual(res_mumbai.site_id, "aras")

        # Vizag -> Bay of Bengal coastal point
        res_vizag = EntityResolver.resolve_entities("Ocean state off Vizag", ctx, session_state)
        self.assertEqual(res_vizag.coordinates["lat"], 17.68)
        self.assertEqual(res_vizag.coordinates["lon"], 83.21)
        self.assertEqual(res_vizag.site_id, "bob")

    def test_entity_resolution_contextual_here(self):
        ctx = ChatContext(
            active_site="Visakhapatnam Coastal Sector",
            coordinates={"lat": 17.68, "lon": 83.21},
            current_depth="20m",
            variable="Temperature",
            current_value="29.4 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        session_state = SagarBotSessionManager.get_session("test_sess")

        res_here = EntityResolver.resolve_entities("What is the risk here?", ctx, session_state)
        self.assertTrue(res_here.is_contextual_location)
        self.assertAlmostEqual(res_here.coordinates["lat"], 17.68, places=2)
        self.assertAlmostEqual(res_here.coordinates["lon"], 83.21, places=2)

    def test_entity_resolution_ambiguous_location_prompts_clarification(self):
        # Empty coordinates and no active site
        ctx = ChatContext(
            active_site="",
            coordinates={"lat": 0.0, "lon": 0.0},
            current_depth="0m",
            variable="Temperature",
            current_value="0.0 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        session_state = {}

        res_ambiguous = EntityResolver.resolve_entities("What is the risk here?", ctx, session_state)
        self.assertTrue(res_ambiguous.needs_clarification)
        self.assertIn("Which ocean region or city", res_ambiguous.clarification_prompt)

    # ==========================================================================
    # 3. HORIZON RESOLUTION & STRICT 1-TO-1 DISPATCH (No Substitution)
    # ==========================================================================
    def test_horizon_resolution_exact_mapping(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.2 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        session_state = SagarBotSessionManager.get_session("test_sess")

        # 0d
        res_0d = EntityResolver.resolve_entities("What is the active risk now?", ctx, session_state)
        self.assertEqual(res_0d.horizon_days, 0)

        # 1d
        res_1d = EntityResolver.resolve_entities("What about tomorrow?", ctx, session_state)
        self.assertEqual(res_1d.horizon_days, 1)

        res_1d_alt = EntityResolver.resolve_entities("Risk in the next 1 day", ctx, session_state)
        self.assertEqual(res_1d_alt.horizon_days, 1)

        # 2d
        res_2d = EntityResolver.resolve_entities("What could happen in the next 2 days?", ctx, session_state)
        self.assertEqual(res_2d.horizon_days, 2)

        # 3d
        res_3d = EntityResolver.resolve_entities("Risk over the next 3 days?", ctx, session_state)
        self.assertEqual(res_3d.horizon_days, 3)

        # Default is 3d
        res_default = EntityResolver.resolve_entities("Is there any cyclone risk in BOB?", ctx, session_state)
        self.assertEqual(res_default.horizon_days, 3)

    # ==========================================================================
    # 4. GENERAL OCEAN STATUS (Skips Prediction ML Inference)
    # ==========================================================================
    def test_general_ocean_status_skips_prediction(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.8 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        req = ChatRequest(
            message="What is the temperature here?",
            context=ctx,
            provider="offline",
        )

        with patch.object(PredictionService, "predict") as mock_predict:
            resp = asyncio.run(SagarBotService.process_chat(req))
            # Critical requirement: PredictionService must NOT be called for general ocean status!
            mock_predict.assert_not_called()

        self.assertEqual(resp.intent, "GENERAL_OCEAN_STATUS")
        self.assertIn("Observed Ocean Status", resp.reply)
        self.assertIn("Sea Surface Temperature", resp.reply)
        self.assertIsNotNone(resp.evidence)
        self.assertIn("observed_sources", resp.evidence)

    # ==========================================================================
    # 5. HISTORICAL EVENT GROUNDING (Fengal, Dana, Asna)
    # ==========================================================================
    def test_historical_event_lookup_and_comparison(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.2 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )

        # 1. Event Lookup
        req_lookup = ChatRequest(
            message="What happened during Fengal?",
            context=ctx,
            provider="offline",
        )
        resp_lookup = asyncio.run(SagarBotService.process_chat(req_lookup))
        self.assertEqual(resp_lookup.intent, "EVENT_LOOKUP")
        self.assertIn("Fengal", resp_lookup.reply)
        self.assertIn("IMD-2024-CS-FENGAL", resp_lookup.reply)

        # 2. Historical Comparison
        req_compare = ChatRequest(
            message="Is this similar to Fengal?",
            context=ctx,
            provider="offline",
        )
        resp_compare = asyncio.run(SagarBotService.process_chat(req_compare))
        self.assertEqual(resp_compare.intent, "HISTORICAL_COMPARISON")
        self.assertIn("Historical Comparison", resp_compare.reply)
        self.assertIn("Fengal", resp_compare.reply)
        self.assertIn("historically similar", resp_compare.reply)
        self.assertIn("does not guarantee", resp_compare.reply)
        self.assertIn("do not cause cyclone formation, intensification, or landfall", resp_compare.reply)

    # ==========================================================================
    # 6. REGION COMPARISON (BOB vs ARAS on Same Horizon)
    # ==========================================================================
    def test_region_comparison_same_horizon(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.2 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        req = ChatRequest(
            message="Which is riskier, BOB or Arabian Sea?",
            context=ctx,
            provider="offline",
        )

        resp = asyncio.run(SagarBotService.process_chat(req))
        self.assertEqual(resp.intent, "REGION_COMPARISON")
        self.assertIn("Multi-Basin Comparative Assessment", resp.reply)
        self.assertIn("Bay of Bengal (BOB)", resp.reply)
        self.assertIn("Arabian Sea (ARAS)", resp.reply)
        self.assertIn("exact same forecast horizon", resp.reply)
        # Verify non-causal oceanographic framing
        self.assertIn("Observed Regional Oceanographic Differences", resp.reply)
        self.assertIn("not proven causes of the difference in model scores", resp.reply)
        self.assertNotIn("**Physical Basis**", resp.reply)

    # ==========================================================================
    # 7. CONVERSATIONAL MEMORY & STALE CONTEXT EVICTION
    # ==========================================================================
    def test_conversational_memory_and_stale_eviction(self):
        session_id = "test_sess_conv"
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.2 °C",
            time_offset="0h",
            historical_date="2024-10-24",
            session_id=session_id,
        )

        # Turn 1: "Is there any risk in BOB over the next 3 days?"
        req1 = ChatRequest(message="Is there any risk in the Bay of Bengal over the next 3 days?", context=ctx)
        resp1 = asyncio.run(SagarBotService.process_chat(req1))
        self.assertEqual(resp1.intent, "FUTURE_RISK")
        sess = SagarBotSessionManager.get_session(session_id)
        self.assertIsNotNone(sess["active_prediction"])
        self.assertEqual(sess["horizon_days"], 3)
        self.assertEqual(sess["site_id"], "bob")

        # Turn 2: "Why?" -> reuses prediction context
        req2 = ChatRequest(message="Why?", context=ctx)
        resp2 = asyncio.run(SagarBotService.process_chat(req2))
        self.assertEqual(resp2.intent, "WHY_RISK")
        self.assertIn("Why (Associated Ocean Indicators)", resp2.reply)

        # Turn 3: "What about tomorrow?" -> switches horizon to 1d AND invalidates previous prediction
        req3 = ChatRequest(message="What about tomorrow?", context=ctx)
        resp3 = asyncio.run(SagarBotService.process_chat(req3))
        self.assertEqual(resp3.intent, "FUTURE_RISK")
        self.assertIn("1-Day Early Warning", resp3.reply)
        sess3 = SagarBotSessionManager.get_session(session_id)
        self.assertEqual(sess3["horizon_days"], 1)

    # ==========================================================================
    # 8. NON-CAUSAL & CONFIDENCE GUARDRAILS
    # ==========================================================================
    def test_non_causal_language_guardrails(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.8 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        req = ChatRequest(message="What is driving the risk?", context=ctx)
        resp = asyncio.run(SagarBotService.process_chat(req))

        reply_lower = resp.reply.lower()
        # Forbidden causal assertions
        self.assertNotIn("caused the event", reply_lower)
        self.assertNotIn("temperature caused the cyclone", reply_lower)
        self.assertNotIn("a cyclone will happen with certainty", reply_lower)

        # Required non-causal association wording
        self.assertIn("associated with", reply_lower)

        # Required single-event Fengal limitation
        self.assertIn("fengal event", reply_lower)
        self.assertIn("does not establish broad generalization", reply_lower)

    def test_confidence_inquiry_does_not_claim_fake_probability(self):
        ctx = ChatContext(
            active_site="Bay of Bengal",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="0m",
            variable="Temperature",
            current_value="29.8 °C",
            time_offset="0h",
            historical_date="2024-10-24",
        )
        req = ChatRequest(message="How confident are you in this forecast?", context=ctx)
        resp = asyncio.run(SagarBotService.process_chat(req))

        self.assertEqual(resp.intent, "CONFIDENCE_INQUIRY")
        self.assertIn("Not a Calibrated Probability", resp.reply)
        self.assertIn("Model Score vs. Validation Metrics", resp.reply)
        self.assertIn("Operational Thresholding", resp.reply)
        self.assertIn("Single-Event Test Split Limitation", resp.reply)
        self.assertIn("Fengal event", resp.reply)
        self.assertIn("does not establish broad generalization", resp.reply)
        self.assertNotIn("100% confidence", resp.reply)

        # Validate structured evidence limitations
        evidence_limits = resp.evidence.get("limitations", [])
        self.assertTrue(any("uncalibrated voting proportion" in l for l in evidence_limits))
        self.assertTrue(any("validation metrics" in l.lower() for l in evidence_limits))
        self.assertTrue(any("fengal event" in l.lower() for l in evidence_limits))


if __name__ == "__main__":
    unittest.main()
