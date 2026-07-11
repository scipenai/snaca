//! Skill registry helpers for SDK users (M1 facade uplift).
//!
//! Re-exports the `snaca-skills` types a host needs to assemble its own skill
//! set and hand it to the engine (via a `SkillTool` in the tool registry).

pub use snaca_skills::{
    DynSkillProvider, LayoutSkillProvider, Skill, SkillError, SkillFrontmatter, SkillProvider,
    SkillRegistry, SkillRegistryBuilder, SkillResult, SkillScope, StaticSkillProvider,
};
