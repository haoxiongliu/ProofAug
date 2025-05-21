"""
Implementation of the proposal structure analysis. 
We do not follow what we do for Isabelle in the ProofAug paper,
since lean tactics can be time-consuming
Rather, we first infer from the proposal the whole structure
(It is like AST tree but for proposal there is no guarantee we can compile)

the body code is like this:

theorem mathd_algebra_114 (a : ℝ) (h₀ : a = 8) :
    (16 * (a ^ 2) ^ ((1 : ℝ) / 3)) ^ ((1 : ℝ) / 3) = 4 := by
  have ha : a ^ 2 = 64 := by
    rw [h₀]
    norm_num
  have h1 : (a ^ 2) ^ ((1 : ℝ) / 3) = 4 := by
    rw [ha]
    have h4 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
      rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
      rw [←Real.rpow_mul]
      norm_num
      all_goals linarith
    exact h4
  have h2 : 16 * (a ^ 2) ^ ((1 : ℝ) / 3) = 64 := by
    rw [h1]
    norm_num
  rw [h2]
  have h3 : (64 : ℝ) ^ ((1 : ℝ) / 3) = 4 := by
    rw [show (64 : ℝ) = 4 ^ (3 : ℝ) by norm_num]
    rw [←Real.rpow_mul]
    norm_num
    all_goals linarith
  exact h3
"""

from __future__ import annotations
from typing import Optional
from prover.utils import remove_lean_comments, statement_starts, analyzable, n_indent
import re
from enum import StrEnum


class BlockState(StrEnum):
    UNVERIFIED = 'unverified'
    WAIT_SORRY = 'wait_sorry'
    SORRY_FAILED = 'sorry_failed'
    PASSED = 'compilation_passed'
    STTM_FAILED = 'sttm_failed'
    COMPLETED = 'completed'

class Snippet(object):
    """A snippet corresponding to a tactic or ends with := by. 
    Always add a newline before adding a new snippet."""
    def __init__(self, content: str = ''):
        self.content = content
        self._proofaug_content = None
    
    @property
    def category(self):
        if statement_starts(self.content):
            return 'statement'
        else:
            return 'normal'
    
    @property
    def proofaug_content(self):
        if self._proofaug_content is None:
            return self.content
        else:
            return self._proofaug_content

    def _receive_snippet(self, snippet: Snippet | str):
        new_content = snippet.content if isinstance(snippet, Snippet) else snippet
        if self.content:
            self.content += "\n" + new_content
        else:
            self.content = new_content

    def __repr__(self):
        return f"-- Snippet(content=\n{self.content})"

class Block(object):
    """each block is a list of snippets and subblocks"""
    def __init__(self, parent: Optional[Block]):
        self.parts = [] # type: list[Block|Snippet]
        self.parent = parent
        self.index = parent.index + f".{len(parent.parts)}" if parent else "0"
        self.level = parent.level + 1 if parent else -1
        self.content_snapshot = None # type: str
        self.state = BlockState.UNVERIFIED # wait_sorry, sorry_failed, verified, sttm_failed
        self._proofaug_parts = None
        
    @property
    def content(self):
        return "\n".join([part.content for part in self.parts])
    
    @property
    def proofaug_parts(self):
        if self._proofaug_parts is None:
            return self.parts
        else:
            return self._proofaug_parts

    @property
    def proofaug_content(self):
        return "\n".join([part.proofaug_content for part in self.proofaug_parts])

    @property
    def category(self):
        return 'block'

    @property
    def statement(self):
        return self.parts[0].content.split(':=')[0]

    def _receive_block(self, block: Block):
        self.parts.append(block)

    def _receive_snippet(self, snippet: Snippet | str, append: bool = False):
        if append and self.parts and isinstance(self.parts[-1], Snippet):
            self.parts[-1]._receive_snippet(snippet)
        else:
            self.parts.append(snippet)
        # legacy code. remain here for reference
        # if self.parts and isinstance(self.parts[-1], Snippet) and self.parts[-1].category == 'statement':
        #     last_is_sttm = True
        # else:
        #     last_is_sttm = False
        # if last_is_sttm or not self.parts or isinstance(self.parts[-1], Block):
        #     self.parts.append(Snippet())
        # self.parts[-1]._receive_snippet(snippet)

    def __repr__(self):
        return f"-- Block(level={self.level}, content=\n{self.content})"




class ProposalStructure(object):
    def __init__(self, proposal: str):
        self.proposal = proposal
        self.root = None
        self._analyze(remove_lean_comments(proposal, normalize=False))

    def _analyze(self, proposal: str):
        lines = proposal.split("\n")
        # determine the blocks by finding 'have' and ':=' and by the indentation
        indent2level = {}
        block_stack = [Block(parent=None)] # type: list[Block]
        i = 0   # pointer
        while i < len(lines):
            line = lines[i]
            if line.strip() == '':
                block_stack[-1]._receive_snippet(Snippet(), append=True)
                i += 1
                continue
            # we assume that the proof never opens a new goal by 'have' when the current same level block is not yet closed
            # determine the level and the current block
            # TODO: reconstruct
            indent = n_indent(line)
            if indent not in indent2level:
                indent2level[indent] = len(indent2level)
            level = indent2level[indent]
            if statement_starts(line):
                for j in range(len(block_stack) - 1, -1, -1):
                    if block_stack[j].level >= level:
                        block_stack.pop()
                    else:
                        break
                last_block = block_stack[-1]
                block = Block(parent=last_block)
                # block._receive_snippet(Snippet(lines[i]))
                sttm_content = line
                while True:
                    i += 1
                    if analyzable(sttm_content) or i >= len(lines) :
                        # the second one corresponding to have xxx := h1.1
                        break
                    if n_indent(lines[i]) < indent:
                        break
                    elif n_indent(lines[i]) == indent:
                        if not lines[i].strip().startswith('|'):
                            break
                    sttm_content += "\n" + lines[i]
                last_block._receive_block(block)
                block_stack.append(block)
                block._receive_snippet(Snippet(sttm_content))

            else:
                for j in range(len(block_stack) - 1, level - 1, -1):
                    if block_stack[j].level >= level:
                        block_stack.pop()
                    else:
                        break    
                block = block_stack[-1]
                tactic_content = line
                i += 1
                while i < len(lines):
                    if n_indent(lines[i]) < indent:
                        break
                    elif n_indent(lines[i]) == indent:
                        special_start_indicators = ['|', '<;>', ')']
                        special_end_indicators = ['|', '<;>', '(']
                        
                        if not ( any(lines[i].strip().startswith(indicator) for indicator in special_start_indicators) \
                            or any(lines[i-1].strip().endswith(indicator) for indicator in special_end_indicators)):
                            break
                    tactic_content += "\n" + lines[i]
                    i += 1
                block._receive_snippet(Snippet(tactic_content))
                
        self.root = block_stack[0]
    
    def _traverse_blocks(self, block: Block):
        # we know that things happen in this block.
        # when error happens, 
        for part in block.parts:
            if isinstance(part, Block):
                self._traverse_blocks(part)
            else:
                print(part.content)

    def traverse(self):
        self._traverse_blocks(self.root)
