// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

contract KDT is ERC20, AccessControl {
    bytes32 public constant ROLE_MINTER = keccak256("MINER");

    constructor() ERC20("Credit", "KDT") {
        // _grantRole(ROLE_MINTER, _treasury);
        // // pre-mint some tokens
        // _mint(_treasury, 1_000_000_000_000);
    }

    // minable to the oracle
    // function mint(address to, uint256 amount) public onlyRole(ROLE_MINTER) {
    //     _mint(to, amount);
    // }
}

contract KDM is ERC20, ERC20Permit, ERC20Votes, AccessControl {
    bytes32 public constant ROLE_MINTER = keccak256("MINTER");

    constructor(address[] memory oracles) ERC20("Creditium", "KDM") ERC20Permit("KDM") {
        for (uint i = 0; i < oracles.length; i++) {
            _grantRole(ROLE_MINTER, oracles[i]);
        }       
    }

    function _afterTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20, ERC20Votes) {
        super._afterTokenTransfer(from, to, amount);
    }

    function _mint(
        address to,
        uint256 amount
    ) internal override(ERC20, ERC20Votes) {
        super._mint(to, amount);
    }

    function _burn(
        address account,
        uint256 amount
    ) internal override(ERC20, ERC20Votes) {
        super._burn(account, amount);
    }

    // minable to the oracle
    function mint(address to, uint256 amount) public onlyRole(ROLE_MINTER) {
        _mint(to, amount);
    }
}
