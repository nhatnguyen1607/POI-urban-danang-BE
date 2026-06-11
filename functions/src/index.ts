import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import { logger } from "firebase-functions";
import { onDocumentCreated } from "firebase-functions/v2/firestore";
import { learnFromFirestoreDocument } from "./onlineLearner";

initializeApp();

const db = getFirestore();

export const learnFromReviewCreated = onDocumentCreated("reviews/{reviewId}", async (event) => {
  const snapshot = event.data;
  if (!snapshot) return;
  const result = await learnFromFirestoreDocument(db, snapshot.data(), "review", event.params.reviewId);
  logger.info("online preference update from review", {
    reviewId: event.params.reviewId,
    ...result,
  });
});

export const learnFromUserAnalyticsCreated = onDocumentCreated("user_analytics/{eventId}", async (event) => {
  const snapshot = event.data;
  if (!snapshot) return;
  const result = await learnFromFirestoreDocument(db, snapshot.data(), "user_analytics", event.params.eventId);
  logger.info("online preference update from user analytics", {
    eventId: event.params.eventId,
    ...result,
  });
});
