import type {
    ISkillStore,
    NewSkillEntry,
    SkillEntry,
    SkillStep,
} from '../types.js';

/**
 * Export a SkillEntry to Markdown with YAML frontmatter.
 */
export function exportSkill(skill: SkillEntry): string {
    const successRate =
        skill.invocationCount > 0
            ? skill.successCount / skill.invocationCount
            : 0;

    const lines: string[] = [];

    // YAML frontmatter
    lines.push('---');
    lines.push(`name: ${skill.name}`);
    lines.push(`description: ${skill.description}`);
    lines.push(`version: ${skill.version}`);
    lines.push(`tags: [${skill.tags.join(', ')}]`);
    lines.push('metrics:');
    lines.push(`  successRate: ${Number(successRate.toFixed(4))}`);
    lines.push(`  invocationCount: ${skill.invocationCount}`);
    lines.push('---');

    // Markdown body — ordered steps
    lines.push('');
    lines.push('## Steps');
    lines.push('');
    for (const step of skill.steps) {
        lines.push(
            `${step.order}. **${step.description}** — Input: ${step.input} → Output: ${step.output}`,
        );
    }
    lines.push('');

    return lines.join('\n');
}

/**
 * Import a Markdown+YAML frontmatter string back into a NewSkillEntry.
 */
export function importSkill(markdown: string): NewSkillEntry {
    const { frontmatter, body } = parseFrontmatter(markdown);

    const name = frontmatter.name ?? 'unnamed';
    const description = frontmatter.description ?? '';
    const tags = parseTags(frontmatter.tags);
    const steps = parseSteps(body);

    return {
        name,
        description,
        steps: steps.length > 0 ? steps : [{ order: 1, description: 'default step', input: 'none', output: 'none' }],
        tags,
    };
}

/**
 * Export all skills from a SkillStore to a Map<filename, markdown>.
 */
export function exportAll(skillStore: ISkillStore): Map<string, string> {
    const result = new Map<string, string>();
    const stats = skillStore.getStats();

    // Get all skill names from version counts
    for (const skillName of Object.keys(stats.versionCounts)) {
        // Get the latest version
        const latestVersion = stats.versionCounts[skillName];
        const entry = skillStore.getVersion(skillName, latestVersion);
        if (entry) {
            const filename = `${sanitizeFilename(entry.name)}.md`;
            result.set(filename, exportSkill(entry));
        }
    }

    return result;
}

// ── Private helpers ─────────────────────────────────────

interface Frontmatter {
    name?: string;
    description?: string;
    version?: string;
    tags?: string;
    [key: string]: string | undefined;
}

function parseFrontmatter(markdown: string): { frontmatter: Frontmatter; body: string } {
    const frontmatter: Frontmatter = {};
    let body = markdown;

    const match = markdown.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/);
    if (match) {
        const yamlBlock = match[1];
        body = match[2];

        // Simple YAML parser for flat key-value pairs (handles nested metrics too)
        for (const line of yamlBlock.split('\n')) {
            const trimmed = line.trim();
            if (trimmed.startsWith('#') || trimmed === '' || trimmed.startsWith('metrics:')) {
                continue;
            }
            // Skip indented metric lines
            if (line.startsWith('  ')) {
                continue;
            }
            const colonIdx = trimmed.indexOf(':');
            if (colonIdx > 0) {
                const key = trimmed.slice(0, colonIdx).trim();
                const value = trimmed.slice(colonIdx + 1).trim();
                frontmatter[key] = value;
            }
        }
    }

    return { frontmatter, body };
}

function parseTags(tagsStr?: string): string[] {
    if (!tagsStr) return [];
    // Parse "[tag1, tag2]" format
    const match = tagsStr.match(/\[(.*)\]/);
    if (match) {
        return match[1]
            .split(',')
            .map((t) => t.trim())
            .filter((t) => t.length > 0);
    }
    return [];
}

function parseSteps(body: string): SkillStep[] {
    const steps: SkillStep[] = [];
    const lines = body.split('\n');

    for (const line of lines) {
        // Match: "1. **description** — Input: xxx → Output: yyy"
        const stepMatch = line.match(
            /^(\d+)\.\s+\*\*(.+?)\*\*\s*[—–-]\s*Input:\s*(.+?)\s*→\s*Output:\s*(.+)$/,
        );
        if (stepMatch) {
            steps.push({
                order: parseInt(stepMatch[1], 10),
                description: stepMatch[2].trim(),
                input: stepMatch[3].trim(),
                output: stepMatch[4].trim(),
            });
        }
    }

    return steps;
}

function sanitizeFilename(name: string): string {
    return name.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
}
