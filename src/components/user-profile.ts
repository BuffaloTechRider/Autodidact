import type Database from 'better-sqlite3';
import type {
    IUserProfile,
    ProfileObservation,
    UserProfileData,
} from '../types.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

interface RawProfileRow {
    name: string;
    preferences: string;
    vocabulary: string;
    conventions: string;
    interaction_count: number;
    created_at: string;
    updated_at: string;
}

export class UserProfile implements IUserProfile {
    private readonly db: Database.Database;
    private readonly logger: Logger;

    constructor(db: Database.Database, logger: Logger = defaultLogger) {
        this.db = db;
        this.logger = logger;
    }

    get(profileName: string): UserProfileData | null {
        const row = this.db
            .prepare(`SELECT * FROM user_profiles WHERE name = ?`)
            .get(profileName) as RawProfileRow | undefined;

        if (!row) {
            return null;
        }
        return this.rowToProfile(row);
    }

    update(profileName: string, observations: ProfileObservation[]): void {
        const existing = this.get(profileName);
        const now = nowISO();

        if (!existing) {
            // Create new profile
            const preferences: Record<string, string> = {};
            const vocabulary: string[] = [];
            const conventions: string[] = [];

            for (const obs of observations) {
                switch (obs.type) {
                    case 'preference':
                        preferences[obs.key] = obs.value;
                        break;
                    case 'vocabulary':
                        if (!vocabulary.includes(obs.value)) {
                            vocabulary.push(obs.value);
                        }
                        break;
                    case 'convention':
                        if (!conventions.includes(obs.value)) {
                            conventions.push(obs.value);
                        }
                        break;
                }
            }

            this.db
                .prepare(
                    `INSERT INTO user_profiles
                    (name, preferences, vocabulary, conventions, interaction_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)`,
                )
                .run(
                    profileName,
                    JSON.stringify(preferences),
                    JSON.stringify(vocabulary),
                    JSON.stringify(conventions),
                    now,
                    now,
                );

            this.logger.debug('UserProfile.update: created new profile', { profileName });
            return;
        }

        // Merge observations additively
        const preferences = { ...existing.preferences };
        const vocabulary = [...existing.vocabulary];
        const conventions = [...existing.conventions];

        for (const obs of observations) {
            switch (obs.type) {
                case 'preference':
                    preferences[obs.key] = obs.value;
                    break;
                case 'vocabulary':
                    if (!vocabulary.includes(obs.value)) {
                        vocabulary.push(obs.value);
                    }
                    break;
                case 'convention':
                    if (!conventions.includes(obs.value)) {
                        conventions.push(obs.value);
                    }
                    break;
            }
        }

        this.db
            .prepare(
                `UPDATE user_profiles
                 SET preferences = ?, vocabulary = ?, conventions = ?,
                     interaction_count = interaction_count + 1, updated_at = ?
                 WHERE name = ?`,
            )
            .run(
                JSON.stringify(preferences),
                JSON.stringify(vocabulary),
                JSON.stringify(conventions),
                now,
                profileName,
            );

        this.logger.debug('UserProfile.update: updated profile', { profileName });
    }

    getContext(profileName: string): string {
        const profile = this.get(profileName);
        if (!profile) {
            return '';
        }

        const parts: string[] = [];

        const prefEntries = Object.entries(profile.preferences);
        if (prefEntries.length > 0) {
            const prefStr = prefEntries.map(([k, v]) => `${k}: ${v}`).join(', ');
            parts.push(`User preferences: ${prefStr}.`);
        }

        if (profile.vocabulary.length > 0) {
            parts.push(`Domain vocabulary: [${profile.vocabulary.join(', ')}].`);
        }

        if (profile.conventions.length > 0) {
            parts.push(`Conventions: ${profile.conventions.join(', ')}.`);
        }

        return parts.join('\n');
    }

    list(): string[] {
        const rows = this.db
            .prepare(`SELECT name FROM user_profiles ORDER BY name`)
            .all() as { name: string }[];

        return rows.map((r) => r.name);
    }

    reset(profileName: string): void {
        this.db
            .prepare(`DELETE FROM user_profiles WHERE name = ?`)
            .run(profileName);

        this.logger.debug('UserProfile.reset: deleted profile', { profileName });
    }

    // ── Private helpers ─────────────────────────────────────

    private rowToProfile(row: RawProfileRow): UserProfileData {
        return {
            name: row.name,
            preferences: JSON.parse(row.preferences) as Record<string, string>,
            vocabulary: JSON.parse(row.vocabulary) as string[],
            conventions: JSON.parse(row.conventions) as string[],
            interactionCount: row.interaction_count,
            createdAt: row.created_at,
            updatedAt: row.updated_at,
        };
    }
}
